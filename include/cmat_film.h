#pragma once
#include "cglobals.h"
#include "crandom.h"
#include "cmaterial.h"
#include "../spectrum.h"
#include <iostream>

static inline float3 filmSampleEtaSpectrum(const Material* a_materials, const Spectrum* a_spectra, const float3 a_wavelengths, uint layer = 0)
{
  float3 eta = float3(a_materials[0].data[FILM_ETA_1 + 2 * layer]);
  if(a_wavelengths.M[0] == 0.0f)
    return eta;

  const uint eta_specId = as_uint(a_materials[0].data[FILM_ETA_1_SPECID + 2 * layer]);

  if(eta_specId < 0xFFFFFFFF)
  {
    eta = SampleSpectrum(a_spectra + eta_specId, a_wavelengths);
    // const uint spectralSamples = uint(sizeof(a_wavelengths.M)/sizeof(a_wavelengths.M[0])); 
    // for(uint i = 0; i < spectralSamples; ++i)
    //   eta[i] = a_spectra[eta_specId].Sample(a_wavelengths[i]);
  }
  
  return eta;
}

static inline float3 filmSampleKSpectrum(const Material* a_materials, const Spectrum* a_spectra, const float3 a_wavelengths, uint layer = 0)
{
  float3 k = float3(a_materials[0].data[FILM_K_1 + 2 * layer]);
  if(a_wavelengths.M[0] == 0.0f)
    return k;

  const uint k_specId = as_uint(a_materials[0].data[FILM_K_1_SPECID + 2 * layer]);

  if(k_specId < 0xFFFFFFFF)
  {
    k = SampleSpectrum(a_spectra + k_specId, a_wavelengths);
    // const uint spectralSamples = uint(sizeof(a_wavelengths.M)/sizeof(a_wavelengths.M[0])); 
    // for(uint i = 0; i < spectralSamples; ++i)
    //   k[i] = a_spectra[k_specId].Sample(a_wavelengths[i]);
  }
  
  return k;
}

static inline void filmSmoothSampleAndEval(const Material* a_materials, const Spectrum* a_spectra, const float3 a_wavelengths,
                                                float4 rands, float3 v, float3 n, float2 tc,
                                                BsdfSample* pRes)
{
  const uint cflags = as_uint(a_materials[0].data[UINT_CFLAGS]);

  float3 eta_1 = filmSampleEtaSpectrum(a_materials, a_spectra, a_wavelengths, 0);
  float3 k_1   = filmSampleKSpectrum(a_materials, a_spectra, a_wavelengths, 0);

  float3 eta_2 = filmSampleEtaSpectrum(a_materials, a_spectra, a_wavelengths, 1);
  float3 k_2   = filmSampleKSpectrum(a_materials, a_spectra, a_wavelengths, 1);

  const float thickness_1 = a_materials[0].data[FILM_THICKNESS_1];

  const float3 pefReflDir = reflect((-1.0f)*v, n);
  const float cosThetaOut = dot(pefReflDir, n);
  float3 dir              = pefReflDir;
  float  pdf              = 1.0f;
  
  float3 val;
  const uint spectralSamples = sizeof(a_wavelengths.M)/sizeof(a_wavelengths.M[0]); 
  for(int i = 0; i < spectralSamples; ++i)
  {
    val[i] = FrFilmRefl(cosThetaOut, complex(1.0f), complex(eta_1[i], k_1[i]), complex(eta_2[i], k_2[i]), thickness_1, a_wavelengths[i]);
    // BSDF is multiplied (outside) by cosThetaOut. For mirrors this shouldn't be done, so we pre-divide here instead
    val[i] = (cosThetaOut <= 1e-6f) ? 0.0f : (val[i] / std::max(cosThetaOut, 1e-6f));  
  }
  
  pRes->val = val; 
  pRes->dir = dir;
  pRes->pdf = pdf;
  pRes->flags = RAY_EVENT_S;
}


static void filmSmoothEval(const Material* a_materials, float3 wavelengths, float3 l, float3 v, float3 n, float2 tc,
                                BsdfEval* pRes)
{
  pRes->val = {0.0f, 0.0f, 0.0f};
  pRes->pdf = 0.0f;
}


static float filmRoughEvalInternal(float3 wo, float3 wi, float3 wm, float2 alpha, complex ior, complex ior2, float thickness, float lambda)
{
  if(wo.z * wi.z < 0) // not in the same hemisphere
  {
    return 0.0f;
  }

  float cosTheta_o = AbsCosTheta(wo);
  float cosTheta_i = AbsCosTheta(wi);
  if (cosTheta_i == 0 || cosTheta_o == 0)
    return 0.0f;

  float F = FrFilmRefl(std::abs(dot(wo, wm)), complex(1.0f), ior, ior2, thickness, lambda);
  float val = trD(wm, alpha) * F * trG(wo, wi, alpha) / (4.0f * cosTheta_i * cosTheta_o);

  return val;
}


static inline void filmRoughSampleAndEval(const Material* a_materials, const Spectrum* a_spectra, const float3 a_wavelengths, 
                                               float4 rands, float3 v, float3 n, float2 tc, float3 alpha_tex, 
                                               BsdfSample* pRes)
{
  if(v.z == 0)
    return;

  const uint cflags = as_uint(a_materials[0].data[UINT_CFLAGS]);

  float3 eta_1 = filmSampleEtaSpectrum(a_materials, a_spectra, a_wavelengths, 0);
  float3 k_1   = filmSampleKSpectrum(a_materials, a_spectra, a_wavelengths, 0);

  float3 eta_2 = filmSampleEtaSpectrum(a_materials, a_spectra, a_wavelengths, 1);
  float3 k_2   = filmSampleKSpectrum(a_materials, a_spectra, a_wavelengths, 1);

  const float thickness_1 = a_materials[0].data[FILM_THICKNESS_1];
  // const uint  texId = as_uint(a_materials[0].data[FILM_TEXID0]);

  const float2 alpha = float2(min(a_materials[0].data[FILM_ROUGH_V], alpha_tex.x), 
                              min(a_materials[0].data[FILM_ROUGH_U], alpha_tex.y));
  // if(texId != 0)
  // {
  //   alpha.x = alpha_tex.x;
  //   alpha.y = alpha_tex.y;
  // }

  float3 nx, ny, nz = n;
  CoordinateSystem(nz, &nx, &ny);
  const float3 wo = float3(dot(v, nx), dot(v, ny), dot(v, nz));
  if(wo.z == 0)
    return;

  if(wo.z == 0)
    return;

  float3 wm = trSample(wo, float2(rands.x, rands.y), alpha);
  float3 wi = reflect((-1.0f) * wo, wm);

  if(wo.z * wi.z < 0) // not in the same hemisphere
  {
    return;
  }

  float3 val;
  const uint spectralSamples = sizeof(a_wavelengths.M)/sizeof(a_wavelengths.M[0]); 
  for(int i = 0; i < spectralSamples; ++i)
  {
    val[i] = filmRoughEvalInternal(wo, wi, wm, alpha, complex(eta_1[i], k_1[i]), complex(eta_2[i], k_2[i]), thickness_1, a_wavelengths[i]);
  }

  pRes->val   = val; 
  pRes->dir   = normalize(wi.x * nx + wi.y * ny + wi.z * nz);
  pRes->pdf   = trPDF(wo, wm, alpha) / (4.0f * std::abs(dot(wo, wm)));
  pRes->flags = RAY_FLAG_HAS_NON_SPEC;
}


static void filmRoughEval(const Material* a_materials, const Spectrum* a_spectra, const float3 a_wavelengths,
                               float3 l, float3 v, float3 n, float2 tc, float3 alpha_tex, 
                               BsdfEval* pRes)
{
  const uint cflags = as_uint(a_materials[0].data[UINT_CFLAGS]);

  float3 eta_1 = filmSampleEtaSpectrum(a_materials, a_spectra, a_wavelengths, 0);
  float3 k_1   = filmSampleKSpectrum(a_materials, a_spectra, a_wavelengths, 0);

  float3 eta_2 = filmSampleEtaSpectrum(a_materials, a_spectra, a_wavelengths, 1);
  float3 k_2   = filmSampleKSpectrum(a_materials, a_spectra, a_wavelengths, 1);

  const float thickness_1 = a_materials[0].data[FILM_THICKNESS_1];

  // const float2 alpha = float2(a_materials[0].data[FILM_ROUGH_V], a_materials[0].data[FILM_ROUGH_U]);
  const float2 alpha = float2(min(a_materials[0].data[FILM_ROUGH_V], alpha_tex.x), 
                              min(a_materials[0].data[FILM_ROUGH_U], alpha_tex.y));

  float3 nx, ny, nz = n;
  CoordinateSystem(nz, &nx, &ny);

  // v = (-1.0f) * v;
  const float3 wo = float3(dot(v, nx), dot(v, ny), dot(v, nz));
  const float3 wi = float3(dot(l, nx), dot(l, ny), dot(l, nz));

  if(wo.z * wi.z < 0.0f)
    return;

  float3 wm = wo + wi;
  if (dot(wm, wm) == 0)
      return;

  wm = normalize(wm);
  float3 val;
  const uint spectralSamples = sizeof(a_wavelengths.M)/sizeof(a_wavelengths.M[0]); 
  for(int i = 0; i < spectralSamples; ++i)
  {
    val[i] = filmRoughEvalInternal(wo, wi, wm, alpha, complex(eta_1[i], k_1[i]), complex(eta_2[i], k_2[i]), thickness_1, a_wavelengths[i]);
  }

  pRes->val = val;

  wm        = FaceForward(wm, float3(0.0f, 0.0f, 1.0f));
  pRes->pdf = trPDF(wo, wm, alpha) / (4.0f * std::abs(dot(wo, wm)));
}

// static inline void filmSampleAndEval(const Material* a_materials, float3 wavelengths, float4 rands, float3 v, float3 n, float2 tc,
//                                           float3 alpha_tex,
//                                           BsdfSample* pRes)
// {
//   const float2 alpha = float2(a_materials[0].data[FILM_ROUGH_V], a_materials[0].data[FILM_ROUGH_U]);

//   if(trEffectivelySmooth(alpha))
//   {
//     filmSmoothSampleAndEval(a_materials, a_spectra, wavelengths, rands, v, n, tc, pRes);
//   }
//   else
//   {
//     filmRoughSampleAndEval(a_materials, a_spectra, wavelengths, rands, v, n, tc, alpha_tex, pRes);
//   }
// }

// static inline void filmEval(const Material* a_materials, float3 wavelengths, float3 l, float3 v, float3 n, float2 tc,
//                                 float3 alpha_tex,
//                                 BsdfEval* pRes)
// {
//   const float2 alpha = float2(a_materials[0].data[FILM_ROUGH_V], a_materials[0].data[FILM_ROUGH_U]);

//   if(trEffectivelySmooth(alpha))
//   {
//     filmSmoothEval(a_materials, wavelengths, l, v, n, tc, pRes);
//   }
//   else
//   {
//     filmRoughEval(a_materials, wavelengths, l, v, n, tc, alpha_tex, pRes);
//   }
// }