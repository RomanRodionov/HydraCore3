#pragma once
#include "cglobals.h"
#include "crandom.h"
#include "cmaterial.h"

struct CMatrix2x2
{
  complex c00, c01, c10, c11;
};

static inline CMatrix2x2 multCMatrix2x2(const CMatrix2x2 m1, const CMatrix2x2 m2)
{
  CMatrix2x2 result;
  result.c00 = complex(m1.c00.re, m1.c00.im) * m2.c00 + complex(m1.c01.re, m1.c01.im) * m2.c10;
  result.c01 = complex(m1.c00.re, m1.c00.im) * m2.c01 + complex(m1.c01.re, m1.c01.im) * m2.c11;
  result.c10 = complex(m1.c10.re, m1.c10.im) * m2.c00 + complex(m1.c11.re, m1.c11.im) * m2.c10;
  result.c11 = complex(m1.c10.re, m1.c10.im) * m2.c01 + complex(m1.c11.re, m1.c11.im) * m2.c11;
  return result;
}

static inline CMatrix2x2 getPropMatrix(complex phaseExp)
{
  CMatrix2x2 prop;
  if (complex_norm(phaseExp) > 1.f)
  {
    prop.c00 = (1.f / (phaseExp * phaseExp));
    prop.c01 = 0.0f;
    prop.c10 = 0.0f;
    prop.c11 = 1.0f;
    return prop;
  }
  else
  {
    prop.c00 = 1.f;
    prop.c01 = 0.0f;
    prop.c10 = 0.0f;
    prop.c11 = phaseExp * phaseExp;
    return prop;
  }
}

static inline complex getPropCoeff(complex phaseExp)
{
  float cnorm = complex_norm(phaseExp);
  if (cnorm > 1.f)
  {
    return cnorm > 1e6f ? complex(1e6f, 0.0f) : phaseExp;
  }
  else
  {
    return cnorm < 1e-6f ? complex(1e6f, 0.0f) : 1 / phaseExp;
  }
}

static inline FrReflRefr TransferMatrixForward(complex cosThetaI, const complex* a_ior, const float* thickness, uint layers, float lambda)
{
  complex sinThetaI = 1.0f - cosThetaI * cosThetaI;
  complex sinThetaF = sinThetaI * a_ior[0].re * a_ior[0].re / (a_ior[1] * a_ior[1]);
  complex cosThetaF = complex_sqrt(1.0f - sinThetaF);
    
  // P polarization
  complex FrRefl = FrComplexRefl(cosThetaI, cosThetaF, a_ior[0], a_ior[1], PolarizationP);
  complex FrRefr = FrComplexRefr(cosThetaI, cosThetaF, a_ior[0], a_ior[1], PolarizationP);

  CMatrix2x2 D_P;
  D_P.c00 = 1.f;
  D_P.c01 = FrRefl;
  D_P.c10 = FrRefl;
  D_P.c11 = 1.f;
  complex coeff_P = FrRefr;

  // S polarization
  FrRefl = FrComplexRefl(cosThetaI, cosThetaF, a_ior[0], a_ior[1], PolarizationS);
  FrRefr = FrComplexRefr(cosThetaI, cosThetaF, a_ior[0], a_ior[1], PolarizationS);

  CMatrix2x2 D_S;
  D_S.c00 = 1.f;
  D_S.c01 = FrRefl;
  D_S.c10 = FrRefl;
  D_S.c11 = 1.f;
  complex coeff_S = FrRefr;

  complex phaseDiff = filmPhaseDiff(cosThetaF, a_ior[1], thickness[0], lambda) * 0.5f;
  complex phaseExp = exp(-phaseDiff.im) * complex(cos(phaseDiff.re), sin(phaseDiff.re));
  CMatrix2x2 P = getPropMatrix(phaseExp);
  complex prop_coeff = getPropCoeff(phaseExp);

  CMatrix2x2 transferMatrix[2] = {multCMatrix2x2(D_P, P), multCMatrix2x2(D_S, P)};
  complex coeff[2] = {coeff_P / prop_coeff, coeff_S / prop_coeff};

  for (int i = 1; i < layers; ++i)
  {
    complex sinThetaT = sinThetaI * a_ior[0].re * a_ior[0].re / (a_ior[i + 1] * a_ior[i + 1]);
    complex cosThetaT = complex_sqrt(1.0f - sinThetaT);
    
    // P polarization
    FrRefl = FrComplexRefl(cosThetaF, cosThetaT, a_ior[i], a_ior[i + 1], PolarizationP);
    FrRefr = FrComplexRefr(cosThetaF, cosThetaT, a_ior[i], a_ior[i + 1], PolarizationP);
 
    D_P.c00 = 1.f;
    D_P.c01 = FrRefl;
    D_P.c10 = FrRefl;
    D_P.c11 = 1.f;
    transferMatrix[0] = multCMatrix2x2(transferMatrix[0], D_P);
    coeff[0] = coeff[0] * FrRefr;

    // S polarization
    FrRefl = FrComplexRefl(cosThetaF, cosThetaT, a_ior[i], a_ior[i + 1], PolarizationS);
    FrRefr = FrComplexRefr(cosThetaF, cosThetaT, a_ior[i], a_ior[i + 1], PolarizationS);
 
    D_S.c00 = 1.f;
    D_S.c01 = FrRefl;
    D_S.c10 = FrRefl;
    D_S.c11 = 1.f;
    transferMatrix[1] = multCMatrix2x2(transferMatrix[1], D_S);
    coeff[1] = coeff[1] * FrRefr;

    if (i < layers - 1)
    {
      phaseDiff = filmPhaseDiff(cosThetaT, a_ior[i + 1], thickness[i], lambda) * 0.5f;
      phaseExp = exp(-phaseDiff.im) * complex(cos(phaseDiff.re), sin(phaseDiff.re));
      P = getPropMatrix(phaseExp);
      prop_coeff = getPropCoeff(phaseExp);
      transferMatrix[0] = multCMatrix2x2(transferMatrix[0], P);
      transferMatrix[1] = multCMatrix2x2(transferMatrix[1], P);
      coeff[0] = coeff[0] / prop_coeff;
      coeff[1] = coeff[1] / prop_coeff;
    }

    sinThetaF = sinThetaT;
    cosThetaF = cosThetaT;
  }

  FrReflRefr result = {0, 0};

  float R_P = complex_norm(transferMatrix[0].c10 / transferMatrix[0].c00);
  float R_S = complex_norm(transferMatrix[1].c10 / transferMatrix[1].c00);
  result.refl = (R_P + R_S) / 2.f;

  float T_P = complex_norm(coeff[0] / transferMatrix[0].c00) * getRefractionFactorP(cosThetaI, cosThetaF, a_ior[0], a_ior[layers]);
  float T_S = complex_norm(coeff[1] / transferMatrix[1].c00) * getRefractionFactorS(cosThetaI, cosThetaF, a_ior[0], a_ior[layers]);
  result.refr = (T_P + T_S) / 2.f;

  return result;
}

static inline FrReflRefr TransferMatrixBackward(complex cosThetaI, const complex* a_ior, const float* thickness, uint layers, float lambda)
{
  complex sinThetaI = 1.0f - cosThetaI * cosThetaI;
  complex sinThetaF = sinThetaI * a_ior[layers].re * a_ior[layers].re / (a_ior[layers - 1] * a_ior[layers - 1]);
  complex cosThetaF = complex_sqrt(1.0f - sinThetaF);
    
  // P polarization
  complex FrRefl = FrComplexRefl(cosThetaI, cosThetaF, a_ior[layers], a_ior[layers - 1], PolarizationP);
  complex FrRefr = FrComplexRefr(cosThetaI, cosThetaF, a_ior[layers], a_ior[layers - 1], PolarizationP);

  CMatrix2x2 D_P;
  D_P.c00 = 1.f;
  D_P.c01 = FrRefl;
  D_P.c10 = FrRefl;
  D_P.c11 = 1.f;
  complex coeff_P = FrRefr;

  // S polarization
  FrRefl = FrComplexRefl(cosThetaI, cosThetaF, a_ior[layers], a_ior[layers - 1], PolarizationS);
  FrRefr = FrComplexRefr(cosThetaI, cosThetaF, a_ior[layers], a_ior[layers - 1], PolarizationS);

  CMatrix2x2 D_S;
  D_S.c00 = 1.f;
  D_S.c01 = FrRefl;
  D_S.c10 = FrRefl;
  D_S.c11 = 1.f;
  complex coeff_S = FrRefr;

  complex phaseDiff = filmPhaseDiff(cosThetaF, a_ior[layers - 1], thickness[layers - 2], lambda) / 2.f;
  complex phaseExp = exp(-phaseDiff.im) * complex(cos(phaseDiff.re), sin(phaseDiff.re));
  CMatrix2x2 P = getPropMatrix(phaseExp);
  complex prop_coeff = getPropCoeff(phaseExp);

  CMatrix2x2 transferMatrix[2] = {multCMatrix2x2(D_P, P), multCMatrix2x2(D_S, P)};
  complex coeff[2] = {coeff_P / prop_coeff, coeff_S / prop_coeff};

  for (int i = 1; i < layers; ++i)
  {
    complex sinThetaT = sinThetaI * a_ior[layers].re * a_ior[layers].re / (a_ior[layers - i - 1] * a_ior[layers - i - 1]);
    complex cosThetaT = complex_sqrt(1.0f - sinThetaT);
    
    // P polarization
    FrRefl = FrComplexRefl(cosThetaF, cosThetaT, a_ior[layers - i], a_ior[layers - i - 1], PolarizationP);
    FrRefr = FrComplexRefr(cosThetaF, cosThetaT, a_ior[layers - i], a_ior[layers - i - 1], PolarizationP);
 
    D_P.c00 = 1.f;
    D_P.c01 = FrRefl;
    D_P.c10 = FrRefl;
    D_P.c11 = 1.f;
    transferMatrix[0] = multCMatrix2x2(transferMatrix[0], D_P);
    coeff[0] = coeff[0] * FrRefr;

    // S polarization
    FrRefl = FrComplexRefl(cosThetaF, cosThetaT, a_ior[layers - i], a_ior[layers - i - 1], PolarizationS);
    FrRefr = FrComplexRefr(cosThetaF, cosThetaT, a_ior[layers - i], a_ior[layers - i - 1], PolarizationS);
 
    D_S.c00 = 1.f;
    D_S.c01 = FrRefl;
    D_S.c10 = FrRefl;
    D_S.c11 = 1.f;
    transferMatrix[1] = multCMatrix2x2(transferMatrix[1], D_S);
    coeff[1] = coeff[1] * FrRefr;

    if (i < layers - 1)
    {
      phaseDiff = filmPhaseDiff(cosThetaT, a_ior[layers - i - 1], thickness[layers - i - 2], lambda) / 2.f;
      phaseExp = exp(-phaseDiff.im) * complex(cos(phaseDiff.re), sin(phaseDiff.re));
      P = getPropMatrix(phaseExp);
      prop_coeff = getPropCoeff(phaseExp);
      transferMatrix[0] = multCMatrix2x2(transferMatrix[0], P);
      transferMatrix[1] = multCMatrix2x2(transferMatrix[1], P);
      coeff[0] = coeff[0] / prop_coeff;
      coeff[1] = coeff[1] / prop_coeff;
    }

    sinThetaF = sinThetaT;
    cosThetaF = cosThetaT;
  }

  FrReflRefr result = {0, 0};

  float R_P = complex_norm(transferMatrix[0].c10 / transferMatrix[0].c00);
  float R_S = complex_norm(transferMatrix[1].c10 / transferMatrix[1].c00);
  result.refl = (R_P + R_S) / 2.f;

  float T_P = complex_norm(coeff[0] / transferMatrix[0].c00) * getRefractionFactorP(cosThetaI, cosThetaF, a_ior[layers], a_ior[0]);
  float T_S = complex_norm(coeff[1] / transferMatrix[1].c00) * getRefractionFactorS(cosThetaI, cosThetaF, a_ior[layers], a_ior[0]);
  result.refr = (T_P + T_S) / 2.f;
  return result;
}
