// ALL SKIP STATEMENTS THAT AFFECT THIS SHADER!!!
// ( $FLASHLIGHT == 0 ) && ( $FLASHLIGHTSHADOWS == 1 )
// ( $FLASHLIGHT == 0 ) && ( $FLASHLIGHTDEPTHFILTERMODE != 0 )
// !$HSV && $HSV_BLEND
// defined $PIXELFOGTYPE && defined $WRITEWATERFOGTODESTALPHA && ( $PIXELFOGTYPE != 1 ) && $WRITEWATERFOGTODESTALPHA
// defined $LIGHTING_PREVIEW && defined $FASTPATHENVMAPTINT && $LIGHTING_PREVIEW && $FASTPATHENVMAPTINT
// defined $LIGHTING_PREVIEW && defined $FASTPATHENVMAPCONTRAST && $LIGHTING_PREVIEW && $FASTPATHENVMAPCONTRAST
// defined $LIGHTING_PREVIEW && defined $FASTPATH && $LIGHTING_PREVIEW && $FASTPATH
// ($FLASHLIGHT || $FLASHLIGHTSHADOWS) && $LIGHTING_PREVIEW
// defined $PIXELFOGTYPE && defined $WRITEWATERFOGTODESTALPHA && ( $PIXELFOGTYPE != 1 ) && $WRITEWATERFOGTODESTALPHA
// defined $LIGHTING_PREVIEW && defined $FASTPATHENVMAPTINT && $LIGHTING_PREVIEW && $FASTPATHENVMAPTINT
// defined $LIGHTING_PREVIEW && defined $FASTPATHENVMAPCONTRAST && $LIGHTING_PREVIEW && $FASTPATHENVMAPCONTRAST
// defined $LIGHTING_PREVIEW && defined $FASTPATH && $LIGHTING_PREVIEW && $FASTPATH
// ($FLASHLIGHT || $FLASHLIGHTSHADOWS) && $LIGHTING_PREVIEW
// defined $PIXELFOGTYPE && defined $WRITEWATERFOGTODESTALPHA && ( $PIXELFOGTYPE != 1 ) && $WRITEWATERFOGTODESTALPHA
// defined $LIGHTING_PREVIEW && defined $FASTPATHENVMAPTINT && $LIGHTING_PREVIEW && $FASTPATHENVMAPTINT
// defined $LIGHTING_PREVIEW && defined $FASTPATHENVMAPCONTRAST && $LIGHTING_PREVIEW && $FASTPATHENVMAPCONTRAST
// defined $LIGHTING_PREVIEW && defined $FASTPATH && $LIGHTING_PREVIEW && $FASTPATH
// ($FLASHLIGHT || $FLASHLIGHTSHADOWS) && $LIGHTING_PREVIEW
// defined $PIXELFOGTYPE && defined $WRITEWATERFOGTODESTALPHA && ( $PIXELFOGTYPE != 1 ) && $WRITEWATERFOGTODESTALPHA
// defined $LIGHTING_PREVIEW && defined $FASTPATHENVMAPTINT && $LIGHTING_PREVIEW && $FASTPATHENVMAPTINT
// defined $LIGHTING_PREVIEW && defined $FASTPATHENVMAPCONTRAST && $LIGHTING_PREVIEW && $FASTPATHENVMAPCONTRAST
// defined $LIGHTING_PREVIEW && defined $FASTPATH && $LIGHTING_PREVIEW && $FASTPATH
// ($FLASHLIGHT || $FLASHLIGHTSHADOWS) && $LIGHTING_PREVIEW

#pragma once
#include "shaderlib/cshader.h"
class pbr_ps20b_Static_Index
{
	unsigned int m_nFLASHLIGHT : 2;
	unsigned int m_nFLASHLIGHTDEPTHFILTERMODE : 2;
	unsigned int m_nLIGHTMAPPED : 2;
	unsigned int m_nEMISSIVE : 2;
	unsigned int m_nPARALLAXOCCLUSION : 1;
	unsigned int m_nWVT : 2;
	unsigned int m_nHSV : 2;
	unsigned int m_nHSV_BLEND : 2;
	unsigned int m_nPARALLAXCORRECT : 2;
	unsigned int m_nBLENDTINTBYMRAOALPHA : 2;
	unsigned int m_nPARALLAXDITHER : 1;
	unsigned int m_nEMISSIVEFRESNEL : 1;
#ifdef _DEBUG
	bool m_bFLASHLIGHT : 1;
	bool m_bFLASHLIGHTDEPTHFILTERMODE : 1;
	bool m_bLIGHTMAPPED : 1;
	bool m_bEMISSIVE : 1;
	bool m_bPARALLAXOCCLUSION : 1;
	bool m_bWVT : 1;
	bool m_bHSV : 1;
	bool m_bHSV_BLEND : 1;
	bool m_bPARALLAXCORRECT : 1;
	bool m_bBLENDTINTBYMRAOALPHA : 1;
	bool m_bPARALLAXDITHER : 1;
	bool m_bEMISSIVEFRESNEL : 1;
#endif	// _DEBUG
public:
	void SetFLASHLIGHT( int i )
	{
		Assert( i >= 0 && i <= 1 );
		m_nFLASHLIGHT = i;
#ifdef _DEBUG
		m_bFLASHLIGHT = true;
#endif	// _DEBUG
	}

	void SetFLASHLIGHTDEPTHFILTERMODE( int i )
	{
		Assert( i >= 0 && i <= 2 );
		m_nFLASHLIGHTDEPTHFILTERMODE = i;
#ifdef _DEBUG
		m_bFLASHLIGHTDEPTHFILTERMODE = true;
#endif	// _DEBUG
	}

	void SetLIGHTMAPPED( int i )
	{
		Assert( i >= 0 && i <= 1 );
		m_nLIGHTMAPPED = i;
#ifdef _DEBUG
		m_bLIGHTMAPPED = true;
#endif	// _DEBUG
	}

	void SetEMISSIVE( int i )
	{
		Assert( i >= 0 && i <= 1 );
		m_nEMISSIVE = i;
#ifdef _DEBUG
		m_bEMISSIVE = true;
#endif	// _DEBUG
	}

	void SetPARALLAXOCCLUSION( int i )
	{
		Assert( i >= 0 && i <= 0 );
		m_nPARALLAXOCCLUSION = i;
#ifdef _DEBUG
		m_bPARALLAXOCCLUSION = true;
#endif	// _DEBUG
	}

	void SetWVT( int i )
	{
		Assert( i >= 0 && i <= 1 );
		m_nWVT = i;
#ifdef _DEBUG
		m_bWVT = true;
#endif	// _DEBUG
	}

	void SetHSV( int i )
	{
		Assert( i >= 0 && i <= 1 );
		m_nHSV = i;
#ifdef _DEBUG
		m_bHSV = true;
#endif	// _DEBUG
	}

	void SetHSV_BLEND( int i )
	{
		Assert( i >= 0 && i <= 1 );
		m_nHSV_BLEND = i;
#ifdef _DEBUG
		m_bHSV_BLEND = true;
#endif	// _DEBUG
	}

	void SetPARALLAXCORRECT( int i )
	{
		Assert( i >= 0 && i <= 1 );
		m_nPARALLAXCORRECT = i;
#ifdef _DEBUG
		m_bPARALLAXCORRECT = true;
#endif	// _DEBUG
	}

	void SetBLENDTINTBYMRAOALPHA( int i )
	{
		Assert( i >= 0 && i <= 1 );
		m_nBLENDTINTBYMRAOALPHA = i;
#ifdef _DEBUG
		m_bBLENDTINTBYMRAOALPHA = true;
#endif	// _DEBUG
	}

	void SetPARALLAXDITHER( int i )
	{
		Assert( i >= 0 && i <= 0 );
		m_nPARALLAXDITHER = i;
#ifdef _DEBUG
		m_bPARALLAXDITHER = true;
#endif	// _DEBUG
	}

	void SetEMISSIVEFRESNEL( int i )
	{
		Assert( i >= 0 && i <= 0 );
		m_nEMISSIVEFRESNEL = i;
#ifdef _DEBUG
		m_bEMISSIVEFRESNEL = true;
#endif	// _DEBUG
	}

	pbr_ps20b_Static_Index( IShaderShadow* pShaderShadow, IMaterialVar** params )
	{
		m_nFLASHLIGHT = 0;
		m_nFLASHLIGHTDEPTHFILTERMODE = 0;
		m_nLIGHTMAPPED = 0;
		m_nEMISSIVE = 0;
		m_nPARALLAXOCCLUSION = 0;
		m_nWVT = 0;
		m_nHSV = 0;
		m_nHSV_BLEND = 0;
		m_nPARALLAXCORRECT = 0;
		m_nBLENDTINTBYMRAOALPHA = 0;
		m_nPARALLAXDITHER = 0;
		m_nEMISSIVEFRESNEL = 0;
#ifdef _DEBUG
		m_bFLASHLIGHT = false;
		m_bFLASHLIGHTDEPTHFILTERMODE = false;
		m_bLIGHTMAPPED = false;
		m_bEMISSIVE = false;
		m_bPARALLAXOCCLUSION = false;
		m_bWVT = false;
		m_bHSV = false;
		m_bHSV_BLEND = false;
		m_bPARALLAXCORRECT = false;
		m_bBLENDTINTBYMRAOALPHA = false;
		m_bPARALLAXDITHER = false;
		m_bEMISSIVEFRESNEL = false;
#endif	// _DEBUG
	}

	int GetIndex() const
	{
		Assert( m_bFLASHLIGHT && m_bFLASHLIGHTDEPTHFILTERMODE && m_bLIGHTMAPPED && m_bEMISSIVE && m_bPARALLAXOCCLUSION && m_bWVT && m_bHSV && m_bHSV_BLEND && m_bPARALLAXCORRECT && m_bBLENDTINTBYMRAOALPHA && m_bPARALLAXDITHER && m_bEMISSIVEFRESNEL );
		AssertMsg( !( ( m_nFLASHLIGHT == 0 ) && ( m_nFLASHLIGHTDEPTHFILTERMODE != 0 ) ), "Invalid combo combination ( ( FLASHLIGHT == 0 ) && ( FLASHLIGHTDEPTHFILTERMODE != 0 ) )" );
		AssertMsg( !( !m_nHSV && m_nHSV_BLEND ), "Invalid combo combination ( !HSV && HSV_BLEND )" );
		return ( 40 * m_nFLASHLIGHT ) + ( 80 * m_nFLASHLIGHTDEPTHFILTERMODE ) + ( 240 * m_nLIGHTMAPPED ) + ( 480 * m_nEMISSIVE ) + ( 960 * m_nPARALLAXOCCLUSION ) + ( 960 * m_nWVT ) + ( 1920 * m_nHSV ) + ( 3840 * m_nHSV_BLEND ) + ( 7680 * m_nPARALLAXCORRECT ) + ( 15360 * m_nBLENDTINTBYMRAOALPHA ) + ( 30720 * m_nPARALLAXDITHER ) + ( 30720 * m_nEMISSIVEFRESNEL ) + 0;
	}
};

#define shaderStaticTest_pbr_ps20b psh_forgot_to_set_static_FLASHLIGHT + psh_forgot_to_set_static_FLASHLIGHTDEPTHFILTERMODE + psh_forgot_to_set_static_LIGHTMAPPED + psh_forgot_to_set_static_EMISSIVE + psh_forgot_to_set_static_PARALLAXOCCLUSION + psh_forgot_to_set_static_WVT + psh_forgot_to_set_static_HSV + psh_forgot_to_set_static_HSV_BLEND + psh_forgot_to_set_static_PARALLAXCORRECT + psh_forgot_to_set_static_BLENDTINTBYMRAOALPHA + psh_forgot_to_set_static_PARALLAXDITHER + psh_forgot_to_set_static_EMISSIVEFRESNEL


class pbr_ps20b_Dynamic_Index
{
	unsigned int m_nNUM_LIGHTS : 3;
	unsigned int m_nWRITE_DEPTH_TO_DESTALPHA : 2;
	unsigned int m_nFLASHLIGHTSHADOWS : 2;
	unsigned int m_nPIXELFOGTYPE : 2;
#ifdef _DEBUG
	bool m_bNUM_LIGHTS : 1;
	bool m_bWRITE_DEPTH_TO_DESTALPHA : 1;
	bool m_bFLASHLIGHTSHADOWS : 1;
#endif	// _DEBUG
public:
	void SetNUM_LIGHTS( int i )
	{
		Assert( i >= 0 && i <= 4 );
		m_nNUM_LIGHTS = i;
#ifdef _DEBUG
		m_bNUM_LIGHTS = true;
#endif	// _DEBUG
	}

	void SetWRITE_DEPTH_TO_DESTALPHA( int i )
	{
		Assert( i >= 0 && i <= 1 );
		m_nWRITE_DEPTH_TO_DESTALPHA = i;
#ifdef _DEBUG
		m_bWRITE_DEPTH_TO_DESTALPHA = true;
#endif	// _DEBUG
	}

	void SetFLASHLIGHTSHADOWS( int i )
	{
		Assert( i >= 0 && i <= 1 );
		m_nFLASHLIGHTSHADOWS = i;
#ifdef _DEBUG
		m_bFLASHLIGHTSHADOWS = true;
#endif	// _DEBUG
	}

	void SetPIXELFOGTYPE( int i )
	{
		Assert( i >= 0 && i <= 1 );
		m_nPIXELFOGTYPE = i;
	}

	pbr_ps20b_Dynamic_Index( IShaderDynamicAPI* pShaderAPI )
	{
		m_nNUM_LIGHTS = 0;
		m_nWRITE_DEPTH_TO_DESTALPHA = 0;
		m_nFLASHLIGHTSHADOWS = 0;
		m_nPIXELFOGTYPE = ( pShaderAPI->GetSceneFogMode() == MATERIAL_FOG_LINEAR_BELOW_FOG_Z ) ;
#ifdef _DEBUG
		m_bNUM_LIGHTS = false;
		m_bWRITE_DEPTH_TO_DESTALPHA = false;
		m_bFLASHLIGHTSHADOWS = false;
#endif	// _DEBUG
	}

	int GetIndex() const
	{
		Assert( m_bNUM_LIGHTS && m_bWRITE_DEPTH_TO_DESTALPHA && m_bFLASHLIGHTSHADOWS );
		return ( 1 * m_nNUM_LIGHTS ) + ( 5 * m_nWRITE_DEPTH_TO_DESTALPHA ) + ( 10 * m_nFLASHLIGHTSHADOWS ) + ( 20 * m_nPIXELFOGTYPE ) + 0;
	}
};

#define shaderDynamicTest_pbr_ps20b psh_forgot_to_set_dynamic_NUM_LIGHTS + psh_forgot_to_set_dynamic_WRITE_DEPTH_TO_DESTALPHA + psh_forgot_to_set_dynamic_FLASHLIGHTSHADOWS
