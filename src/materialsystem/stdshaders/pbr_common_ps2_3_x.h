#define PI			3.141592653589793
#define EPSILON		0.00001

// PBR BRDF functions
// Based on https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf

// Single term for separable Schlick-GGX
float GaSchlickG1( float cosTheta, float k )
{
	return cosTheta / ( cosTheta * ( 1.0 - k ) + k );
}

// Schlick-GGX approximation of geometric attenuation function using Smith's method
float GaSchlickGGX( float cosLi, float cosLo, float roughness )
{
	// k is alpha/2, to better fit the Smith model for GGX
	//
	// Substituting you get:
	//  alpha = roughness*roughness
	//  k = alpha/2 = roughness*roughness / 2

	float k = ( roughness * roughness ) * 0.5;
	return GaSchlickG1( cosLi, k ) * GaSchlickG1( cosLo, k );
}

// Schlick-GGX approximation of geometric attenuation function using Smith's method
// This version remaps the roughness to reduce "hotness", however this should only be used for analytical lights
float GaSchlickGGXRemapped( float cosLi, float cosLo, float roughness )
{
	// k is alpha/2, to better fit the Smith model for GGX
	// Roughness is also remapped using (roughness + 1)/2 before squaring
	//
	// Substituting the remapping, you get:
	//  alpha = ((roughness+1)/2)^2 = (roughness+1)*(roughness+1)/4
	//  k = alpha/2 = (roughness+1)*(roughness+1)/8

	float r = roughness + 1.0;
	float k = ( r * r ) / 8.0;
	return GaSchlickG1( cosLi, k ) * GaSchlickG1( cosLo, k );
}

// Shlick's approximation of the Fresnel factor
float3 FresnelSchlick( float3 F0, float cosTheta )
{
	return F0 + ( 1.0 - F0 ) * pow( 1.0 - cosTheta, 5.0 );
}

// Shlick's approximation of the Fresnel factor with account for roughness
float3 FresnelSchlickRoughness( float3 F0, float cosTheta, float roughness )
{
	return F0 + max( 0.0, ( 1.0 - roughness ) - F0 ) * pow( 1.0 - cosTheta, 5.0 );
}

// GGX/Towbridge-Reitz normal distribution function
// Uses Disney's reparametrization of alpha = roughness^2
float NdfGGX( float cosLh, float roughness )
{
	float alpha = roughness * roughness;
	float alphaSq = alpha * alpha;

	float denom = cosLh * ( cosLh * alphaSq - cosLh ) + 1.0;
	return alphaSq / ( PI * denom * denom );
}


// Monte Carlo integration, approximate analytic version based on Dimitar Lazarov's work
// https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
float2 EnvBRDFApprox(float3 SpecularColor, float Roughness, float NoV)
{
	const float4 c0 = { -1, -0.0275, -0.572, 0.022 };
	const float4 c1 = { 1, 0.0425, 1.04, -0.04 };
	float4 r = Roughness * c0 + c1;
	float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
	float2 AB = float2(-1.04, 1.04) * a004 + r.zw;
	return AB;
}

// Compute the matrix used to transform tangent space normals to world space
// This expects DirectX normal maps in Mikk Tangent Space http://www.mikktspace.com
float3x3 ComputeTangentFrame(float3 N, float3 P, float2 uv, out float3 T, out float3 B, out float sign_det)
{
	float3 dp1 = ddx(P);
	float3 dp2 = ddy(P);
	float2 duv1 = ddx(uv);
	float2 duv2 = ddy(uv);

	sign_det = dot(dp2, cross(N, dp1)) > 0.0 ? -1 : 1;

	float3x3 M = float3x3(dp1, dp2, cross(dp1, dp2));
	float2x3 inverseM = float2x3(cross(M[1], M[2]), cross(M[2], M[0]));
	T = normalize(mul(float2(duv1.x, duv2.x), inverseM));
	B = normalize(mul(float2(duv1.y, duv2.y), inverseM));
	return float3x3(T, B, N);
}

float GetAttenForLight(float4 lightAtten, int lightNum)
{
	return lightAtten[lightNum];
}

// Calculate direct light for one source
float3 CalculateLight(float3 lightIn, float3 lightIntensity, float3 lightOut, float3 normal, float3 fresnelReflectance, float roughness, float metalness, float lightDirectionAngle, float3 albedo)
{
	// Lh
	float3 HalfAngle = normalize(lightIn + lightOut); 
	float cosLightIn = max(0.0, dot(normal, lightIn));
	float cosHalfAngle = max(0.0, dot(normal, HalfAngle));

	// F - Calculate Fresnel term for direct lighting
	float3 F = FresnelSchlick(fresnelReflectance, max(0.0, dot(HalfAngle, lightOut)));
	float3 F2 = FresnelSchlick(fresnelReflectance, max(0.0, dot(normal, lightOut)));
	float3 F3 = FresnelSchlick(fresnelReflectance, max(0.0, dot(normal, lightIn)));

	// D - Calculate normal distribution for specular BRDF
	float D = NdfGGX(cosHalfAngle, roughness);

	// Calculate geometric attenuation for specular BRDF
	float G = GaSchlickGGXRemapped(cosLightIn, lightDirectionAngle, roughness);

	// Cook-Torrance specular microfacet BRDF
	float3 specularBRDF = (F * D * G) / max(EPSILON, 4.0 * cosLightIn * lightDirectionAngle);

#if LIGHTMAPPED && !FLASHLIGHT

	// Ambient light from static lights is already precomputed in the lightmap. Don't add it again
	return specularBRDF * lightIntensity * cosLightIn;

#else

	// Diffuse scattering happens due to light being refracted multiple times by a dielectric medium
	// Metals on the other hand either reflect or absorb energso diffuse contribution is always, zero
	// To be energy conserving we must scale diffuse BRDF contribution based on Fresnel factor & metalness
	float3 kd = lerp((float3(1, 1, 1) - F2 ) * (float3(1, 1, 1) - F3 ), float3(0, 0, 0), metalness);
	float3 diffuseBRDF = kd * albedo;
	return (diffuseBRDF + specularBRDF) * lightIntensity * cosLightIn;

#endif // LIGHTMAPPED && !FLASHLIGHT
}

// Get diffuse ambient light
float3 AmbientLookupLightmap(float3 worldPos,
	float3 normal,
	float3 textureNormal,
	float4 lightmapTexCoord1And2,
	float4 lightmapTexCoord3,
	sampler LightmapSampler,
	float4 modulation,
	float flShadow)
{
	float2 bumpCoord1;
	float2 bumpCoord2;
	float2 bumpCoord3;

	ComputeBumpedLightmapCoordinates(
			lightmapTexCoord1And2, lightmapTexCoord3.xy,
			bumpCoord1, bumpCoord2, bumpCoord3);

	float3 lightmapColor1 = LightMapSample(LightmapSampler, bumpCoord1);
	float3 lightmapColor2 = LightMapSample(LightmapSampler, bumpCoord2);
	float3 lightmapColor3 = LightMapSample(LightmapSampler, bumpCoord3);

	float3 dp;
	dp.x = saturate(dot(textureNormal, bumpBasis[0]));
	dp.y = saturate(dot(textureNormal, bumpBasis[1]));
	dp.z = saturate(dot(textureNormal, bumpBasis[2]));
	dp *= dp;

	float3 diffuseLighting = dp.x * lightmapColor1.rgb +
							 dp.y * lightmapColor2.rgb +
							 dp.z * lightmapColor3.rgb;

	float sum = dot(dp, float3(1, 1, 1));
	if ( sum != 0 )
		diffuseLighting *= modulation.xyz / sum;
	return diffuseLighting;
}

float3 AmbientLookup(float3 worldPos, float3 normal, float3 ambientCube[6], float3 textureNormal, float4 lightmapTexCoord1And2, float4 lightmapTexCoord3, sampler LightmapSampler, float4 modulation, float flShadow)
{
	#if ( LIGHTMAPPED )
	{
		return AmbientLookupLightmap(worldPos,
			normal,
			textureNormal,
			lightmapTexCoord1And2,
			lightmapTexCoord3,
			LightmapSampler,
			modulation,
			flShadow);
	}
	#else
	{
		return PixelShaderAmbientLight(normal, ambientCube);
	}
	#endif
}

float ScreenSpaceBayerDither( float2 vScreenPos )
{
	int x = vScreenPos.x % 8;
	int y = vScreenPos.y % 8;

	const int dither[8][8] = {
	{ 0, 32, 8, 40, 2, 34, 10, 42}, /* 8x8 Bayer ordered dithering */
	{48, 16, 56, 24, 50, 18, 58, 26}, /* pattern. Each input pixel */
	{12, 44, 4, 36, 14, 46, 6, 38}, /* is scaled to the 0..63 range */
	{60, 28, 52, 20, 62, 30, 54, 22}, /* before looking in this table */
	{ 3, 35, 11, 43, 1, 33, 9, 41}, /* to determine the action. */
	{51, 19, 59, 27, 49, 17, 57, 25},
	{15, 47, 7, 39, 13, 45, 5, 37},
	{63, 31, 55, 23, 61, 29, 53, 21} };

	float limit = 0.0;

	{
		limit = (dither[x][y]+1)/64.0;
	}

	return limit;
}

float3 WorldToRelative(float3 worldVector, float3 surfTangent, float3 surfBasis, float3 surfNormal)
{
	return float3(
		dot(worldVector, surfTangent),
		dot(worldVector, surfBasis),
		dot(worldVector, surfNormal)
	);
}

float3 rgb2hsv(float3 c)
{
	float4 P = (c.g < c.b) ? float4(c.bg, -1.0, 2.0 / 3.0) : float4(c.gb, 0.0, -1.0 / 3.0);
	float4 Q = (c.r < P.x) ? float4(P.xyw, c.r) : float4(c.r, P.yzx);
	float C = Q.x - min(Q.w, Q.y);
	return float3(abs(Q.z + (Q.w - Q.y) / (6.0 * C + EPSILON)), C / (Q.x + EPSILON), Q.x);
}

float3 hsv2rgb(float3 c)
{
	float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
	return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
}
