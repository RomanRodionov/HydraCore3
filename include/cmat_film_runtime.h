#pragma once
#include "cglobals.h"
#include "crandom.h"
#include "cmaterial.h"
#include "../spectrum.h"
#include <iostream>
#include "airy_reflectance.h"

#ifndef KERNEL_SLICER
#define KSPEC_FILMS_STACK_SIZE Integrator::KSPEC_FILMS_STACK_SIZE
#endif

// struct IORVector
// {
//   complex value[KSPEC_FILMS_STACK_SIZE];
// };

static inline void filmSmoothSampleAndEval(const Material* a_materials, 
        // const complex *a_ior, 
        // const IORVector a_ior,
        const complex a_ior[KSPEC_FILMS_STACK_SIZE],
        const float* thickness,
        uint layers, const float4 a_wavelengths, const float _extIOR, float4 rands, float3 v, float3 n, float2 tc, BsdfSample* pRes)
{

  const float extIOR = a_materials[0].data[FILM_ETA_EXT];
  
  if ((pRes->flags & RAY_FLAG_HAS_INV_NORMAL) != 0) // inside of object
  {
    n = -1 * n;
  }
  bool reversed = dot(n, v) < 0.f && a_ior[layers].im < 0.001;

  float3 s, t = n;
  CoordinateSystemV2(n, &s, &t);
  float3 wi = float3(dot(v, s), dot(v, t), dot(v, n));

  float cosThetaI = clamp(fabs(wi.z), 0.001, 1.0f);
  float ior = a_ior[layers].re / extIOR;
  float4 R = float4(0.0f), T = float4(0.0f);

  for(uint32_t k = 0; k < SPECTRUM_SAMPLE_SZ && a_wavelengths[k] > 0.0f; ++k)
  {
    FrReflRefr result = {0.f, 0.f};
    if (layers == 2)
    {
      if (!reversed)
      {
        result = FrFilm(cosThetaI, a_ior[0], a_ior[1], a_ior[2], thickness[0], a_wavelengths[k]);
      }
      else
      {
        result = FrFilm(cosThetaI, a_ior[2], a_ior[1], a_ior[0], thickness[0], a_wavelengths[k]);
      }
    }
    else if (layers > 2)
    {
      if (!reversed)
      { 
        //result = multFrFilm(cosThetaI, a_ior, thickness, layers, a_wavelengths[k]);
        float sinThetaI = 1.0f - cosThetaI * cosThetaI;
        complex sinThetaO = sinThetaI * a_ior[0].re * a_ior[0].re / (complex(a_ior[layers].re, a_ior[layers].im) * a_ior[layers]);
        complex cosThetaO = complex_sqrt(1.0f - sinThetaO);
        complex sinTheta = sinThetaI * a_ior[0].re * a_ior[0].re / (complex(a_ior[layers - 1].re, a_ior[layers - 1].im) * a_ior[layers - 1]);
        complex cosTheta = complex_sqrt(1.0f - sinTheta);
        complex FrReflS = FrComplexRefl(cosTheta, cosThetaO, a_ior[layers - 1], a_ior[layers], PolarizationS);
        complex FrReflP = FrComplexRefl(cosTheta, cosThetaO, a_ior[layers - 1], a_ior[layers], PolarizationP);
        complex FrRefrS = FrComplexRefr(cosTheta, cosThetaO, a_ior[layers - 1], a_ior[layers], PolarizationS);
        complex FrRefrP = FrComplexRefr(cosTheta, cosThetaO, a_ior[layers - 1], a_ior[layers], PolarizationP);
        for (uint i = layers - 1; i > 0; --i)
        {
          complex cosThetaOld = cosTheta;
          sinTheta = sinThetaI * a_ior[0].re * a_ior[0].re / (complex(a_ior[i - 1].re, a_ior[i - 1].im) * a_ior[i - 1]);
          cosTheta = complex_sqrt(1.0f - sinTheta);

          complex phaseDiff = filmPhaseDiff(cosThetaOld, a_ior[i], thickness[i - 1], a_wavelengths[k]);
          complex temp_exp = exp(-phaseDiff.im / 2.f) * complex(cos(phaseDiff.re / 2.f), sin(phaseDiff.re / 2.f));

          complex FrReflI = FrComplexRefl(cosTheta, cosThetaOld, a_ior[i - 1], a_ior[i], PolarizationS);
          complex FrRefrI = FrComplexRefr(cosTheta, cosThetaOld, a_ior[i - 1], a_ior[i], PolarizationS);
          FrRefrS = FrRefrI * FrRefrS * temp_exp;
          FrReflS = FrReflS * temp_exp * temp_exp;
          complex denom = 1.f / (1 + FrReflI * FrReflS);
          FrRefrS = FrRefrS * denom;
          FrReflS = (FrReflI + FrReflS) * denom;

          FrReflI = FrComplexRefl(cosTheta, cosThetaOld, a_ior[i - 1], a_ior[i], PolarizationP);
          FrRefrI = FrComplexRefr(cosTheta, cosThetaOld, a_ior[i - 1], a_ior[i], PolarizationP);
          FrRefrP = FrRefrI * FrRefrP * temp_exp;
          FrReflP = FrReflP * temp_exp * temp_exp;
          denom = 1.f / (1 + FrReflI * FrReflP);
          FrRefrP = FrRefrP * denom;
          FrReflP = (FrReflI + FrReflP) * denom;
        }
        result.refl = (complex_norm(FrReflS) + complex_norm(FrReflP)) * 0.5f;
        result.refr = (complex_norm(FrRefrS) + complex_norm(FrRefrP)) * 0.5f;
        result.refr *= getRefractionFactor(cosThetaI, cosThetaO, a_ior[0], a_ior[layers]);
      }
      else
      {
        //result = multFrFilm_r(cosThetaI, a_ior, thickness, layers, a_wavelengths[k]);
        float sinThetaI = 1.0f - cosThetaI * cosThetaI;

        complex sinThetaO = sinThetaI * a_ior[layers].re * a_ior[layers].re / (complex(a_ior[0].re, a_ior[0].im) * a_ior[0]);
        complex cosThetaO = complex_sqrt(1.0f - sinThetaO);
        complex sinTheta = sinThetaI * a_ior[layers].re * a_ior[layers].re / (complex(a_ior[1].re, a_ior[1].im) * a_ior[1]);
        complex cosTheta = complex_sqrt(1.0f - sinTheta);
        complex FrReflS = FrComplexRefl(cosTheta, cosThetaO, a_ior[1], a_ior[0], PolarizationS);
        complex FrReflP = FrComplexRefl(cosTheta, cosThetaO, a_ior[1], a_ior[0], PolarizationP);
        complex FrRefrS = FrComplexRefr(cosTheta, cosThetaO, a_ior[1], a_ior[0], PolarizationS);
        complex FrRefrP = FrComplexRefr(cosTheta, cosThetaO, a_ior[1], a_ior[0], PolarizationP);
        for (uint i = 1; i < layers; ++i)
        {
          complex cosThetaOld = cosTheta;
          complex sinTheta = sinThetaI * a_ior[layers].re * a_ior[layers].re / (complex(a_ior[i + 1].re, a_ior[i + 1].im) * a_ior[i + 1]);
          complex cosTheta = complex_sqrt(1.0f - sinTheta);
          
          complex phaseDiff = filmPhaseDiff(cosThetaOld, a_ior[i], thickness[i - 1], a_wavelengths[k]);
          complex temp_exp = exp(-phaseDiff.im / 2.f) * complex(cos(phaseDiff.re / 2.f), sin(phaseDiff.re / 2.f));

          complex FrReflI = FrComplexRefl(cosTheta, cosThetaOld, a_ior[i + 1], a_ior[i], PolarizationS);
          complex FrRefrI = FrComplexRefr(cosTheta, cosThetaOld, a_ior[i + 1], a_ior[i], PolarizationS);
          FrRefrS = FrRefrI * FrRefrS * temp_exp;
          FrReflS = FrReflS * temp_exp * temp_exp;
          complex denom = 1.f / (1 + FrReflI * FrReflS);
          FrRefrS = FrRefrS * denom;
          FrReflS = (FrReflI + FrReflS) * denom;

          FrReflI = FrComplexRefl(cosTheta, cosThetaOld, a_ior[i + 1], a_ior[i], PolarizationP);
          FrRefrI = FrComplexRefr(cosTheta, cosThetaOld, a_ior[i + 1], a_ior[i], PolarizationP);
          FrRefrP = FrRefrI * FrRefrP * temp_exp;
          FrReflP = FrReflP * temp_exp * temp_exp;
          denom = 1.f / (1 + FrReflI * FrReflP);
          FrRefrP = FrRefrP * denom;
          FrReflP = (FrReflI + FrReflP) * denom;
        }
        result.refl = (complex_norm(FrReflS) + complex_norm(FrReflP)) * 0.5;
        result.refr = (complex_norm(FrRefrS) + complex_norm(FrRefrP)) * 0.5;
        result.refr *= getRefractionFactor(cosThetaI, cosThetaO, a_ior[layers], a_ior[0]);
      }
    }
    R[k] = result.refl;
    T[k] = result.refr;
  }

  if (a_ior[layers].im > 0.001)
  {
    float3 wo = float3(-wi.x, -wi.y, wi.z);
    pRes->val = R;
    pRes->pdf = 1.f;
    pRes->dir = normalize(wo.x * s + wo.y * t + wo.z * n);
    pRes->flags |= RAY_EVENT_S;
    pRes->ior = _extIOR;
  }
  else
  {
    if (rands.x * (sum(R) + sum(T)) < sum(R))
    {
      float3 wo = float3(-wi.x, -wi.y, wi.z);
      pRes->val = R;
      pRes->pdf = sum(R) / (sum(R) + sum(T));
      pRes->dir = normalize(wo.x * s + wo.y * t + wo.z * n);
      pRes->flags |= RAY_EVENT_S;
      pRes->ior = _extIOR;
    }
    else
    {
      float4 fr = FrDielectricDetailedV2(wi.z, ior);
      const float cosThetaT = fr.y;
      const float eta_ti = fr.w;  

      float3 wo = refract(wi, cosThetaT, eta_ti);
      pRes->val = T;
      pRes->pdf = sum(T) / (sum(R) + sum(T));
      pRes->dir = normalize(wo.x * s + wo.y * t + wo.z * n);
      pRes->flags |= (RAY_EVENT_S | RAY_EVENT_T);
      pRes->ior = (_extIOR == a_ior[layers].re) ? extIOR : a_ior[layers].re;
    }
  }

  pRes->val /= std::max(std::abs(dot(pRes->dir, n)), 1e-6f);
}

static void filmSmoothEval(const Material* a_materials, const float4 eta_1, const float4 k_1, const float4 eta_2, const float4 k_2, float4 wavelengths, float3 l, float3 v, float3 n, float2 tc,
                                BsdfEval* pRes)
{
  pRes->val = {0.0f, 0.0f, 0.0f, 0.0f};
  pRes->pdf = 0.0f;
}

static inline void filmRoughSampleAndEval(const Material* a_materials, 
        // const complex *a_ior, 
        // const IORVector a_ior,
        const complex a_ior[KSPEC_FILMS_STACK_SIZE],
        const float* thickness,
        uint layers, const float4 a_wavelengths, const float _extIOR, float4 rands, float3 v, float3 n, float2 tc, float3 alpha_tex, BsdfSample* pRes)
{
  const float extIOR = a_materials[0].data[FILM_ETA_EXT];

  if ((pRes->flags & RAY_FLAG_HAS_INV_NORMAL) != 0) // inside of object
  {
    n = -1 * n;
  }

  bool reversed = dot(v, n) < 0.f && a_ior[layers].im < 0.001;

  const float2 alpha = float2(min(a_materials[0].data[FILM_ROUGH_V], alpha_tex.x), 
                              min(a_materials[0].data[FILM_ROUGH_U], alpha_tex.y));

  float3 s, t = n;
  CoordinateSystemV2(n, &s, &t);
  float3 wi = float3(dot(v, s), dot(v, t), dot(v, n));

  float ior = a_ior[layers].re / extIOR;
  if (reversed)
  {
    wi = -1 * wi;
    ior = 1.f / ior;
  }

  const float4 wm_pdf = sample_visible_normal(wi, {rands.x, rands.y}, alpha);
  const float3 wm = to_float3(wm_pdf);
  if(wm_pdf.w == 0.0f) // not in the same hemisphere
  {
    return;
  }

  float cosThetaI = clamp(fabs(dot(wi, wm)), 0.001, 1.0f);
  float4 R = float4(0.0f), T = float4(0.0f);

  for(uint32_t k = 0; k < SPECTRUM_SAMPLE_SZ && a_wavelengths[k] > 0.0f; ++k)
  {
    FrReflRefr result = {0.f, 0.f};

    if (layers == 2)
    {
      if (!reversed)
      {
        //result = FrFilm(cosThetaI, a_ior[0], a_ior[1], a_ior[2], 50.f + (n.y + 1.f) * 100.f, a_wavelengths[k]);
        result = FrFilm(cosThetaI, a_ior[0], a_ior[1], a_ior[2], thickness[0], a_wavelengths[k]);
      }
      else
      {
        result = FrFilm(cosThetaI, a_ior[2], a_ior[1], a_ior[0], thickness[0], a_wavelengths[k]);
      }
    }
    else if (layers > 2)
    {
      if (!reversed)
      { 
        //result = multFrFilm(cosThetaI, a_ior, thickness, layers, a_wavelengths[k]);
        complex a_cosTheta[KSPEC_FILMS_STACK_SIZE + 1];
        complex a_phaseDiff[KSPEC_FILMS_STACK_SIZE - 1];
        a_cosTheta[0] = complex(cosThetaI);

        float sinThetaI = 1.0f - cosThetaI * cosThetaI;
        complex sinTheta;
        complex cosTheta;
        for (uint i = 1; i <= layers; ++i)
        {
          sinTheta = sinThetaI * a_ior[0].re * a_ior[0].re / (complex(a_ior[i].re, a_ior[i].im) * a_ior[i]);
          cosTheta = complex_sqrt(1.0f - sinTheta);
          a_cosTheta[i] = cosTheta;
          if (i < layers)
            a_phaseDiff[i - 1] = filmPhaseDiff(cosTheta, a_ior[i], thickness[i - 1], a_wavelengths[k]);
        }
        uint polarization[2] = {PolarizationS, PolarizationP};
        for (uint p = 0; p < 2; ++p)
        {
          complex FrRefl = FrComplexRefl(a_cosTheta[layers - 1], a_cosTheta[layers], a_ior[layers - 1], a_ior[layers], p);
          complex FrRefr = FrComplexRefr(a_cosTheta[layers - 1], a_cosTheta[layers], a_ior[layers - 1], a_ior[layers], p);
          for (uint i = layers - 1; i > 0; --i)
          {
            complex FrReflI = FrComplexRefl(a_cosTheta[i - 1], a_cosTheta[i], a_ior[i - 1], a_ior[i], p);
            complex FrRefrI = FrComplexRefr(a_cosTheta[i - 1], a_cosTheta[i], a_ior[i - 1], a_ior[i], p);
            complex temp_exp = exp(-a_phaseDiff[i - 1].im / 2.f) * complex(cos(a_phaseDiff[i - 1].re / 2.f), sin(a_phaseDiff[i - 1].re / 2.f));
            FrRefr = FrRefrI * FrRefr * temp_exp;
            FrRefl = FrRefl * temp_exp * temp_exp;
            complex denom = 1.f / (1 + FrReflI * FrRefl);
            FrRefr = FrRefr * denom;
            FrRefl = (FrReflI + FrRefl) * denom;
          }
          result.refl += complex_norm(FrRefl) / 2;
          result.refr += complex_norm(FrRefr) / 2;
        }
        result.refr *= getRefractionFactor(cosThetaI, a_cosTheta[layers], a_ior[0], a_ior[layers]);
      }
      else
      {
        //result = multFrFilm_r(cosThetaI, a_ior, thickness, layers, a_wavelengths[k]);
        complex a_cosTheta[KSPEC_FILMS_STACK_SIZE + 1];
        complex a_phaseDiff[KSPEC_FILMS_STACK_SIZE - 1];
        a_cosTheta[layers] = complex(cosThetaI);

        float sinThetaI = 1.0f - cosThetaI * cosThetaI;
        complex sinTheta = complex(1.0);
        complex cosTheta = complex(1.0);
        for (uint i = layers; i > 0; --i)
        {
          sinTheta = sinThetaI * a_ior[layers].re * a_ior[layers].re / (complex(a_ior[i - 1].re, a_ior[i - 1].im) * a_ior[i - 1]);
          cosTheta = complex_sqrt(1.0f - sinTheta);
          a_cosTheta[i - 1] = cosTheta;
          if (i > 1)
            a_phaseDiff[i - 2] = filmPhaseDiff(cosTheta, a_ior[i - 1], thickness[i - 2], a_wavelengths[k]);
        }
        uint polarization[2] = {PolarizationS, PolarizationP};
        for (uint p = 0; p < 2; ++p)
        {
          complex FrRefl = FrComplexRefl(a_cosTheta[1], a_cosTheta[0], a_ior[1], a_ior[0], p);
          complex FrRefr = FrComplexRefr(a_cosTheta[1], a_cosTheta[0], a_ior[1], a_ior[0], p);
          for (uint i = 1; i < layers; ++i)
          {
            complex FrReflI = FrComplexRefl(a_cosTheta[i + 1], a_cosTheta[i], a_ior[i + 1], a_ior[i], p);
            complex FrRefrI = FrComplexRefr(a_cosTheta[i + 1], a_cosTheta[i], a_ior[i + 1], a_ior[i], p);
            complex temp_exp = exp(-a_phaseDiff[i - 1].im / 2.f) * complex(cos(a_phaseDiff[i - 1].re / 2.f), sin(a_phaseDiff[i - 1].re / 2.f));
            FrRefr = FrRefrI * FrRefr * temp_exp;
            FrRefl = FrRefl * temp_exp * temp_exp;
            complex denom = 1.f / (1 + FrReflI * FrRefl);
            FrRefr = FrRefr * denom;
            FrRefl = (FrReflI + FrRefl) * denom;
          }
          result.refl += complex_norm(FrRefl) / 2;
          result.refr += complex_norm(FrRefr) / 2;
        }
        result.refr *= getRefractionFactor(cosThetaI, a_cosTheta[0], a_ior[layers], a_ior[0]);
      }
    } 

    R[k] = result.refl;
    T[k] = result.refr;
  }

  if (a_ior[layers].im > 0.001)
  {
    float3 wo = reflect((-1.0f) * wi, wm);
    if (wi.z < 0.f || wo.z <= 0.f)
    {
      return;
    }
    const float cos_theta_i = std::max(wi.z, EPSILON_32);
    const float cos_theta_o = std::max(wo.z, EPSILON_32);
    pRes->pdf = trPDF(wi, wm, alpha) / (4.0f * std::abs(dot(wi, wm)));
    pRes->val = trD(wm, alpha) * microfacet_G(wi, wo, wm, alpha) * R / (4.0f * cos_theta_i * cos_theta_o);
    if (reversed)
    {
      wo = -1 * wo;
    }
    pRes->dir = normalize(wo.x * s + wo.y * t + wo.z * n);
    pRes->flags = RAY_FLAG_HAS_NON_SPEC;
    pRes->ior = _extIOR;
  }
  else
  {
    if (rands.w * (sum(R) + sum(T)) < sum(R))
    {
      float3 wo = reflect((-1.0f) * wi, wm);
      if (wi.z < 0.f || wo.z <= 0.f)
      {
        return;
      }
      const float cos_theta_i = std::max(wi.z, EPSILON_32);
      const float cos_theta_o = std::max(wo.z, EPSILON_32);
      pRes->pdf = trPDF(wi, wm, alpha) / (4.0f * std::abs(dot(wi, wm))) * sum(R) / (sum(R) + sum(T));
      pRes->val = trD(wm, alpha) * microfacet_G(wi, wo, wm, alpha) * R / (4.0f * cos_theta_i * cos_theta_o);
      if (reversed)
      {
        wo = -1 * wo;
      }
      pRes->dir = normalize(wo.x * s + wo.y * t + wo.z * n);
      pRes->flags = RAY_FLAG_HAS_NON_SPEC;
      pRes->ior = _extIOR;
    }
    else
    {
      float4 fr = FrDielectricDetailedV2(dot(wi, wm), ior);
      const float cosThetaT = fr.y;
      const float eta_it = fr.z;
      const float eta_ti = fr.w;  

      float3 ws, wt;
      CoordinateSystemV2(wm, &ws, &wt);
      const float3 local_wi = {dot(ws, wi), dot(wt, wi), dot(wm, wi)};
      const float3 local_wo = refract(local_wi, cosThetaT, eta_ti);
      float3 wo = local_wo.x * ws + local_wo.y * wt + local_wo.z * wm;
      if (wo.z > 0.f)
      {
        return;
      }
      const float cos_theta_i = std::max(wi.z, EPSILON_32);
      const float cos_theta_o = std::max(wo.z, EPSILON_32);
      if (fabs(eta_it - 1.f) <= 1e-6f)
      {
        pRes->pdf = trPDF(wi, wm, alpha) / (4.0f * std::abs(dot(wi, wm))) * sum(T) / (sum(R) + sum(T));
        pRes->val = trD(wm, alpha) * microfacet_G(wi, wo, wm, alpha) * T / (4.0f * -cos_theta_i * cos_theta_o);
      }
      else
      {
        float denom = sqr(dot(wo, wm) + dot(wi, wm) / eta_it);
        float dwm_dwi = fabs(dot(wo, wm)) / denom;
        pRes->pdf = trPDF(wi, wm, alpha) * dwm_dwi * sum(T) / (sum(R) + sum(T));
        pRes->val = trD(wm, alpha) * microfacet_G(wi, wo, wm, alpha) * T * fabs(dot(wi, wm) * dot(wo, wm) / (cos_theta_i * cos_theta_o * denom));
      }
      if (reversed)
      {
        wo = -1 * wo;
      }
      pRes->dir = normalize(wo.x * s + wo.y * t + wo.z * n);
      pRes->flags = RAY_FLAG_HAS_NON_SPEC;
      pRes->ior = (_extIOR == a_ior[layers].re) ? extIOR : a_ior[layers].re;
    }
  }
}


static void filmRoughEval(const Material* a_materials, 
        // const complex *a_ior, 
        // IORVector a_ior,
        const complex a_ior[KSPEC_FILMS_STACK_SIZE],
        const float* thickness,
        uint layers, const float4 a_wavelengths, float3 l, float3 v, float3 n, float2 tc, float3 alpha_tex, BsdfEval* pRes)
{
  if (a_ior[layers].im < 0.001)
  {
    return;
  }

  const float extIOR = a_materials[0].data[FILM_ETA_EXT];
  uint32_t refl_offset;
  uint32_t refr_offset;

  bool reversed = dot(v, n) < 0.f && a_ior[layers].im < 0.001;

  const float2 alpha = float2(min(a_materials[0].data[FILM_ROUGH_V], alpha_tex.x), 
                              min(a_materials[0].data[FILM_ROUGH_U], alpha_tex.y));

  float3 s, t = n;
  CoordinateSystemV2(n, &s, &t);
  const float3 wo = float3(dot(l, s), dot(l, t), dot(l, n));
  const float3 wi = float3(dot(v, s), dot(v, t), dot(v, n));
  const float3 wm = normalize(wo + wi);

  if (wi.z * wo.z < 0.f)
  {
    return;
  }

  float ior = a_ior[layers].re / extIOR;
  if (reversed)
  {
    ior = 1.f / ior;
  }

  float cosThetaI = clamp(fabs(dot(wo, wm)), 0.001, 1.0f);
  
  float4 R = float4(0.0f);
  
  for(uint32_t k = 0; k < SPECTRUM_SAMPLE_SZ && a_wavelengths[k] > 0.0f; ++k)
  {
    if (layers == 2)
    {
      if (!reversed)
      {
        //R = FrFilmRefl(cosThetaI, a_ior[0], a_ior[1], a_ior[2], 50.f + (n.y + 1.f) * 100.f, a_wavelengths[k]);
        R[k] = FrFilmRefl(cosThetaI, a_ior[0], a_ior[1], a_ior[2], thickness[0], a_wavelengths[k]);
      }
      else
      {
        R[k] = FrFilmRefl(cosThetaI, a_ior[2], a_ior[1], a_ior[0], thickness[0], a_wavelengths[k]); 
      }
    }
    else if (layers > 2)
    {
      if (!reversed)
      { 
        //result = multFrFilm(cosThetaI, a_ior, thickness, layers, a_wavelengths[k]);
        complex a_cosTheta[KSPEC_FILMS_STACK_SIZE + 1];
        complex a_phaseDiff[KSPEC_FILMS_STACK_SIZE - 1];
        a_cosTheta[0] = complex(cosThetaI);

        float sinThetaI = 1.0f - cosThetaI * cosThetaI;
        complex sinTheta;
        complex cosTheta;
        for (uint i = 1; i <= layers; ++i)
        {
          sinTheta = sinThetaI * a_ior[0].re * a_ior[0].re / (complex(a_ior[i].re, a_ior[i].im) * a_ior[i]);
          cosTheta = complex_sqrt(1.0f - sinTheta);
          a_cosTheta[i] = cosTheta;
          if (i < layers)
            a_phaseDiff[i - 1] = filmPhaseDiff(cosTheta, a_ior[i], thickness[i - 1], a_wavelengths[k]);
        }
        uint polarization[2] = {PolarizationS, PolarizationP};
        for (uint p = 0; p < 2; ++p)
        {
          complex FrRefl = FrComplexRefl(a_cosTheta[layers - 1], a_cosTheta[layers], a_ior[layers - 1], a_ior[layers], p);
          for (uint i = layers - 1; i > 0; --i)
          {
            complex FrReflI = FrComplexRefl(a_cosTheta[i - 1], a_cosTheta[i], a_ior[i - 1], a_ior[i], p);
            FrRefl = FrRefl * exp(-a_phaseDiff[i - 1].im) * complex(cos(a_phaseDiff[i - 1].re), sin(a_phaseDiff[i - 1].re));
            complex denom = 1.f / (1 + FrReflI * FrRefl);
            FrRefl = (FrReflI + FrRefl) * denom;
          }
          R[k] += complex_norm(FrRefl) / 2;
        }
      }
      else
      {
        //result = multFrFilm_r(cosThetaI, a_ior, thickness, layers, a_wavelengths[k]);
        complex a_cosTheta[KSPEC_FILMS_STACK_SIZE + 1];
        complex a_phaseDiff[KSPEC_FILMS_STACK_SIZE - 1];
        a_cosTheta[layers] = complex(cosThetaI);

        float sinThetaI = 1.0f - cosThetaI * cosThetaI;
        complex sinTheta = complex(1.0);
        complex cosTheta = complex(1.0);
        for (uint i = layers; i > 0; --i)
        {
          sinTheta = sinThetaI * a_ior[layers].re * a_ior[layers].re / (complex(a_ior[i - 1].re, a_ior[i - 1].im) * a_ior[i - 1]);
          cosTheta = complex_sqrt(1.0f - sinTheta);
          a_cosTheta[i - 1] = cosTheta;
          if (i > 1)
            a_phaseDiff[i - 2] = filmPhaseDiff(cosTheta, a_ior[i - 1], thickness[i - 2], a_wavelengths[k]);
        }
        uint polarization[2] = {PolarizationS, PolarizationP};
        for (uint p = 0; p < 2; ++p)
        {
          complex FrRefl = FrComplexRefl(a_cosTheta[1], a_cosTheta[0], a_ior[1], a_ior[0], p);
          for (uint i = 1; i < layers; ++i)
          {
            complex FrReflI = FrComplexRefl(a_cosTheta[i + 1], a_cosTheta[i], a_ior[i + 1], a_ior[i], p);
            FrRefl = FrRefl * exp(-a_phaseDiff[i - 1].im) * complex(cos(a_phaseDiff[i - 1].re), sin(a_phaseDiff[i - 1].re));
            complex denom = 1.f / (1 + FrReflI * FrRefl);
            FrRefl = (FrReflI + FrRefl) * denom;
          }
          R[k] += complex_norm(FrRefl) / 2;
        }
      }
    }
  }

  const float cos_theta_i = std::max(wi.z, EPSILON_32);
  const float cos_theta_o = std::max(wo.z, EPSILON_32);

  float D = eval_microfacet(wm, alpha, 1);
  float G = microfacet_G(wi, wo, wm, alpha);
  pRes->pdf = trPDF(wi, wm, alpha) / (4.0f * std::abs(dot(wi, wm)));
  pRes->val = trD(wm, alpha) * microfacet_G(wi, wo, wm, alpha) * R / (4.0f * cos_theta_i * cos_theta_o);
}