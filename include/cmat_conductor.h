#pragma once
#include "cglobals.h"
#include "crandom.h"
#include "cmaterial.h"


static inline void conductorSmoothSampleAndEval(const GLTFMaterial* a_materials, float4 rands, float3 v, float3 n, float2 tc, 
                                                float3 color,
                                                BsdfSample* pRes)
{
  const uint  cflags  = a_materials[0].cflags;
  const float eta     = a_materials[0].metalColor[2];
  const float k       = a_materials[0].metalColor[3];
  
  const float3 pefReflDir = reflect((-1.0f)*v, n);
  const float cosThetaOut = dot(pefReflDir, n);
  float3 dir = pefReflDir;
  float  pdf = 1.0f;
  float  val = FrComplexConductor(cosThetaOut, complex{eta, k});
  
  val = (cosThetaOut <= 1e-6f) ? 0.0f : (val / std::max(cosThetaOut, 1e-6f));  // BSDF is multiplied (outside) by cosThetaOut. For mirrors this shouldn't be done, so we pre-divide here instead.

  pRes->val = float3(val, val, val); 
  pRes->dir = dir;
  pRes->pdf = pdf;
  pRes->flags = RAY_EVENT_S;
}


static void conductorSmoothEval(const GLTFMaterial* a_materials, float3 l, float3 v, float3 n, float2 tc, 
                                float3 color, 
                                BsdfEval* pRes)
{
  pRes->color = {0.0f, 0.0f, 0.0f};
  pRes->pdf = 0.0f;
}

static inline void conductorRoughSampleAndEval(const GLTFMaterial* a_materials, float4 rands, float3 v, float3 n, float2 tc, 
                                                float3 color,
                                                BsdfSample* pRes)
{
  if(v.z == 0)
    return;

  const uint  cflags  = a_materials[0].cflags;
  const float eta     = a_materials[0].metalColor[2];
  const float k       = a_materials[0].metalColor[3];
  const float2 alpha  = float2(a_materials[0].metalColor[0], a_materials[0].metalColor[1]);

  float3 nx, ny, nz = n;
  CoordinateSystem(nz, &nx, &ny);
  const float3 wo = float3(dot(v, nx), dot(v, ny), dot(v, nz));

  float3 wm = trSample(wo, float2(rands.x, rands.y), alpha);
  float3 wi = reflect((-1.0f) * wo, wm);

  if(wo.z * wi.z < 0) // not in the same hemisphere
  {
    return;
  }

  float pdf = trPDF(wo, wm, alpha) / (4.0f * std::abs(dot(wo, wm)));

  float cosTheta_o = AbsCosTheta(wo);
  float cosTheta_i = AbsCosTheta(wi);
  if (cosTheta_i == 0 || cosTheta_o == 0)
      return;

  float F = FrComplexConductor(std::abs(dot(wo, wm)), complex{eta, k});
  float val = trD(wm, alpha) * F * trG(wo, wi, alpha) / (4.0f * cosTheta_i * cosTheta_o);

  pRes->val = float3(val, val, val); 
  pRes->dir = normalize(wi.x * nx + wi.y * ny + wi.z * nz);
  pRes->pdf = pdf;
  pRes->flags = RAY_EVENT_S;
}


static void conductorRoughEval(const GLTFMaterial* a_materials, float3 l, float3 v, float3 n, float2 tc, 
                                float3 color, 
                                BsdfEval* pRes)
{
  const uint  cflags  = a_materials[0].cflags;
  const float eta     = a_materials[0].metalColor[2];
  const float k       = a_materials[0].metalColor[3];
  const float2 alpha  = float2(a_materials[0].metalColor[0], a_materials[0].metalColor[1]);

  float3 nx, ny, nz = n;
  CoordinateSystem(nz, &nx, &ny);
  const float3 wo = float3(dot(v, nx), dot(v, ny), dot(v, nz));
  const float3 wi = float3(dot(l, nx), dot(l, ny), dot(l, nz));

  if(wo.z * wi.z < 0.0f)
    return;

  float cosTheta_o = AbsCosTheta(wo);
  float cosTheta_i = AbsCosTheta(wi); 
  // float cosTheta_o = std::abs(dot(n, v));
  // float cosTheta_i = std::abs(dot(n, l)); 
  if (cosTheta_i == 0 || cosTheta_o == 0)
    return; 

  float3 wm = wo + wi;
  if (dot(wm, wm) == 0)
      return;

  wm = normalize(wm);

  float F = FrComplexConductor(std::abs(dot(wo, wm)), complex{eta, k});
  float val = trD(wm, alpha) * F * trG(wo, wi, alpha) / (4.0f * cosTheta_i * cosTheta_o);

  pRes->color = float3(val, val, val); 
  pRes->pdf = trPDF(wo, wm, alpha) / (4.0f * std::abs(dot(wo, wm)));
}

static inline void conductorSampleAndEval(const GLTFMaterial* a_materials, float4 rands, float3 v, float3 n, float2 tc, 
                                          float3 color,
                                          BsdfSample* pRes)
{
  const float2 alpha  = float2(a_materials[0].metalColor[0], a_materials[0].metalColor[1]);
  if(trEffectivelySmooth(alpha))
  {
    conductorSmoothSampleAndEval(a_materials, rands, v, n, tc, color, pRes);
  }
  else
  {
    conductorRoughSampleAndEval(a_materials, rands, v, n, tc, color, pRes);
  }
}

static inline void conductorEval(const GLTFMaterial* a_materials, float3 l, float3 v, float3 n, float2 tc, 
                                float3 color, 
                                BsdfEval* pRes)
{
  const float2 alpha  = float2(a_materials[0].metalColor[0], a_materials[0].metalColor[1]);
  if(trEffectivelySmooth(alpha))
  {
    conductorSmoothEval(a_materials, l, v, n, tc, color, pRes);
  }
  else
  {
    conductorRoughEval(a_materials, l, v, n, tc, color, pRes);
  }
}