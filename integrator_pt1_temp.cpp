////////////////////////////////////////////////////
//// input file: /home/roman/Desktop/hydra3spectr/kernel_slicer/../HydraCore3/integrator_pt1.cpp
////////////////////////////////////////////////////
#include "integrator_pt.h"
#include "include/crandom.h"

#include <chrono>
#include <string>

#include "Image2d.h"
using LiteImage::Image2D;
using LiteImage::Sampler;
using LiteImage::ICombinedImageSampler;
using namespace LiteMath;

void Integrator::InitRandomGens(int a_maxThreads)
{
  m_randomGens.resize(a_maxThreads);
  #pragma omp parallel for default(shared)
  for(int i=0;i<a_maxThreads;i++)
    m_randomGens[i] = RandomGenInit(i);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void Integrator::kernel_InitEyeRay2(uint tid, const uint* packedXY, 
                                   float4* rayPosAndNear, float4* rayDirAndFar, float3* wavelengths, 
                                   float4* accumColor,    float4* accumuThoroughput,
                                   RandomGen* gen, uint* rayFlags, MisData* misData) // 
{
  *accumColor        = make_float4(0,0,0,1);
  *accumuThoroughput = make_float4(1,1,1,1);
  RandomGen genLocal = m_randomGens[tid];
  *rayFlags          = 0;
  *misData           = makeInitialMisData();

  const uint XY = packedXY[tid];

  const uint x = (XY & 0x0000FFFF);
  const uint y = (XY & 0xFFFF0000) >> 16;
  const float2 pixelOffsets = rndFloat2_Pseudo(&genLocal);

  float3 rayDir = EyeRayDirNormalized((float(x) + pixelOffsets.x)/float(m_winWidth), 
                                      (float(y) + pixelOffsets.y)/float(m_winHeight), m_projInv);
  float3 rayPos = float3(0,0,0);

  transform_ray3f(m_worldViewInv, &rayPos, &rayDir);

  if(m_spectral_mode)
  {
    float u = rndFloat1_Pseudo(&genLocal);
    *wavelengths = SampleWavelengths(u);
  }
  else
  {
    *wavelengths = {0.0f, 0.0f, 0.0f};
  }

  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, FLT_MAX);
  *gen           = genLocal;
}


void Integrator::kernel_RayTrace2(uint tid, const float4* rayPosAndNear, const float4* rayDirAndFar,
                                 float4* out_hit1, float4* out_hit2, uint* out_instId, uint* rayFlags)
{
  uint currRayFlags = *rayFlags;
  if(isDeadRay(currRayFlags))
    return;
    
  const float4 rayPos = *rayPosAndNear;
  const float4 rayDir = *rayDirAndFar ;

  const CRT_Hit hit   = m_pAccelStruct->RayQuery_NearestHit(rayPos, rayDir);

  if(hit.geomId != uint32_t(-1))
  {
    const float2 uv       = float2(hit.coords[0], hit.coords[1]);
    const float3 hitPos   = to_float3(rayPos) + (hit.t*0.999999f)*to_float3(rayDir); // set hit slightlyt closer to old ray origin to prevent self-interseaction and e.t.c bugs

    const uint triOffset  = m_matIdOffsets[hit.geomId];
    const uint vertOffset = m_vertOffset  [hit.geomId];
  
    const uint A = m_triIndices[(triOffset + hit.primId)*3 + 0];
    const uint B = m_triIndices[(triOffset + hit.primId)*3 + 1];
    const uint C = m_triIndices[(triOffset + hit.primId)*3 + 2];
  
    const float3 A_norm = to_float3(m_vNorm4f[A + vertOffset]);
    const float3 B_norm = to_float3(m_vNorm4f[B + vertOffset]);
    const float3 C_norm = to_float3(m_vNorm4f[C + vertOffset]);

    const float2 A_texc = m_vTexc2f[A + vertOffset];
    const float2 B_texc = m_vTexc2f[B + vertOffset];
    const float2 C_texc = m_vTexc2f[C + vertOffset];
      
    float3 hitNorm     = (1.0f - uv.x - uv.y)*A_norm + uv.y*B_norm + uv.x*C_norm;
    float2 hitTexCoord = (1.0f - uv.x - uv.y)*A_texc + uv.y*B_texc + uv.x*C_texc;
  
    // transform surface point with matrix and flip normal if needed
    //
    hitNorm                = normalize(mul3x3(m_normMatrices[hit.instId], hitNorm));
    const float flipNorm   = dot(to_float3(rayDir), hitNorm) > 0.001f ? -1.0f : 1.0f; // beware of transparent materials which use normal sign to identity "inside/outside" glass for example
    hitNorm                = flipNorm * hitNorm;
    
    if (flipNorm < 0.0f) currRayFlags |=  RAY_FLAG_HAS_INV_NORMAL;
    else                 currRayFlags &= ~RAY_FLAG_HAS_INV_NORMAL;

    const uint midOriginal = m_matIdByPrimId[m_matIdOffsets[hit.geomId] + hit.primId];
    const uint midRemaped  = RemapMaterialId(midOriginal, hit.instId);

    *rayFlags              = packMatId(currRayFlags, midRemaped);
    *out_hit1              = to_float4(hitPos,  hitTexCoord.x); 
    *out_hit2              = to_float4(hitNorm, hitTexCoord.y);
    *out_instId            = hit.instId;
  }
  else
    *rayFlags              = currRayFlags | (RAY_FLAG_IS_DEAD | RAY_FLAG_OUT_OF_SCENE);
}

float3 Integrator::GetLightSourceIntensity(uint a_lightId, const float3* a_wavelengths)
{
  float3 lightColor = to_float3(m_lights[a_lightId].intensity);
  if(!m_spectral_mode)
    return lightColor;

  const uint specId = as_uint(m_lights[a_lightId].ids.x);

  if(specId < 0xFFFFFFFF)
  {
    lightColor = SampleSpectrum(m_spectra.data() + specId, *a_wavelengths);
    // const uint spectralSamples = uint(sizeof(a_wavelengths->M) / sizeof(a_wavelengths->M[0])); 
    // for(uint i = 0; i < spectralSamples; ++i)
    //   lightColor[i] = m_spectra[specId].Sample(a_wavelengths->M[i]);
  }
  return lightColor;
}


void Integrator::kernel_SampleLightSource(uint tid, const float4* rayPosAndNear, const float4* rayDirAndFar, 
                                          const float3* wavelengths, const float4* in_hitPart1, const float4* in_hitPart2, 
                                          const uint* rayFlags,  
                                          RandomGen* a_gen, float4* out_shadeColor)
{
  const uint currRayFlags = *rayFlags;
  if(isDeadRay(currRayFlags))
    return;
    
  const uint32_t matId = extractMatId(currRayFlags);
  const float3 ray_dir = to_float3(*rayDirAndFar);
  
  const float4 data1  = *in_hitPart1;
  const float4 data2  = *in_hitPart2;
  const float3 lambda = *wavelengths;

  SurfaceHit hit;
  hit.pos  = to_float3(data1);
  hit.norm = to_float3(data2);
  hit.uv   = float2(data1.w, data2.w);

  const float2 rands = rndFloat2_Pseudo(a_gen); // don't use single rndFloat4 (!!!)
  const float rndId  = rndFloat1_Pseudo(a_gen); // don't use single rndFloat4 (!!!)
  const int lightId  = int(std::floor(rndId * float(m_lights.size())));
  
  const LightSample lSam = LightSampleRev(lightId, rands, hit.pos);
  const float  hitDist   = std::sqrt(dot(hit.pos - lSam.pos, hit.pos - lSam.pos));

  const float3 shadowRayDir = normalize(lSam.pos - hit.pos); // explicitSam.direction;
  const float3 shadowRayPos = hit.pos + hit.norm*std::max(maxcomp(hit.pos), 1.0f)*5e-6f; // TODO: see Ray Tracing Gems, also use flatNormal for offset
  const bool   inShadow     = m_pAccelStruct->RayQuery_AnyHit(to_float4(shadowRayPos, 0.0f), to_float4(shadowRayDir, hitDist*0.9995f));
  
  if(!inShadow && dot(shadowRayDir, lSam.norm) < 0.0f) 
  {
    const BsdfEval bsdfV    = MaterialEval(matId, lambda, shadowRayDir, (-1.0f)*ray_dir, hit.norm, hit.uv);
    const float cosThetaOut = std::max(dot(shadowRayDir, hit.norm), 0.0f);
    
    float      lgtPdfW      = LightPdfSelectRev(lightId) * LightEvalPDF(lightId, shadowRayPos, shadowRayDir, lSam.pos, lSam.norm);
    float      misWeight    = (m_intergatorType == INTEGRATOR_MIS_PT) ? misWeightHeuristic(lgtPdfW, bsdfV.pdf) : 1.0f;
    const bool isDirect     = (m_lights[lightId].geomType == LIGHT_GEOM_DIRECT); 
    
    if(isDirect)
    {
      misWeight = 1.0f;
      lgtPdfW   = 1.0f;
    }
    
    const float3 lightColor = GetLightSourceIntensity(lightId, wavelengths);
    *out_shadeColor = to_float4((lightColor * bsdfV.val / lgtPdfW) * cosThetaOut * misWeight, 0.0f);
  }
  else
    *out_shadeColor = float4(0.0f, 0.0f, 0.0f, 1.0f);
}

void Integrator::kernel_NextBounce(uint tid, uint bounce, const float4* in_hitPart1, const float4* in_hitPart2, const uint* in_instId,
                                   const float4* in_shadeColor, float4* rayPosAndNear, float4* rayDirAndFar, const float3* wavelengths,
                                   float4* accumColor, float4* accumThoroughput, RandomGen* a_gen, MisData* misPrev, uint* rayFlags)
{
  const uint currRayFlags = *rayFlags;
  if(isDeadRay(currRayFlags))
    return;
    
  const uint32_t matId = extractMatId(currRayFlags);

  // process surface hit case
  //
  const float3 ray_dir = to_float3(*rayDirAndFar);
  const float3 ray_pos = to_float3(*rayPosAndNear);
  const float3 lambda  = *wavelengths;
  
  const float4 data1 = *in_hitPart1;
  const float4 data2 = *in_hitPart2;
  
  SurfaceHit hit;
  hit.pos  = to_float3(data1);
  hit.norm = to_float3(data2);
  hit.uv   = float2(data1.w, data2.w);
  
  const MisData prevBounce = *misPrev;
  const float   prevPdfW   = prevBounce.matSamplePdf;
  const float   prevPdfA   = (prevPdfW >= 0.0f) ? PdfWtoA(prevPdfW, length(ray_pos - hit.norm), prevBounce.cosTheta) : 1.0f;

  // process light hit case
  //
  if(as_uint(m_materials[matId].data[UINT_MTYPE]) == MAT_TYPE_LIGHT_SOURCE)
  {
    const uint texId       = as_uint(m_materials[matId].data[EMISSION_TEXID0]);
    const float2 texCoordT = mulRows2x4(m_materials[matId].row0[0], m_materials[matId].row1[0], hit.uv);
    const float3 texColor  = to_float3(m_textures[texId]->sample(texCoordT));
    float3 lightColor      = to_float3(m_materials[matId].colors[EMISSION_COLOR]);

    if(m_spectral_mode)
    {
      const uint specId = as_uint(m_materials[matId].data[EMISSION_SPECID0]);
      if(specId < 0xFFFFFFFF)
      {
        lightColor = SampleSpectrum(m_spectra.data() + specId, *wavelengths);
      }
    }

    const float3 lightIntensity = lightColor * texColor;
    const uint lightId          = m_instIdToLightInstId[*in_instId]; //m_materials[matId].data[UINT_LIGHTID];
    
    float lightCos = 1.0f;
    float lightDirectionAtten = 1.0f;
    if(lightId != 0xFFFFFFFF)
    {
      lightCos = dot(to_float3(*rayDirAndFar), to_float3(m_lights[lightId].norm));
      lightDirectionAtten = (lightCos < 0.0f || m_lights[lightId].geomType == LIGHT_GEOM_SPHERE) ? 1.0f : 0.0f;
    }

    float misWeight = 1.0f;
    if(m_intergatorType == INTEGRATOR_MIS_PT) 
    {
      if(bounce > 0)
      {
        if(lightId != 0xFFFFFFFF)
        {
          const float lgtPdf  = LightPdfSelectRev(lightId) * LightEvalPDF(lightId, ray_pos, ray_dir, hit.pos, hit.norm);
          misWeight           = misWeightHeuristic(prevPdfW, lgtPdf);
          if (prevPdfW <= 0.0f) // specular bounce
            misWeight = 1.0f;
        }
      }
    }
    else if(m_intergatorType == INTEGRATOR_SHADOW_PT && hasNonSpecular(currRayFlags))
      misWeight = 0.0f;
    
    float4 currAccumColor      = *accumColor;
    float4 currAccumThroughput = *accumThoroughput;
    
    currAccumColor.x += currAccumThroughput.x * lightIntensity.x * misWeight * lightDirectionAtten;
    currAccumColor.y += currAccumThroughput.y * lightIntensity.y * misWeight * lightDirectionAtten;
    currAccumColor.z += currAccumThroughput.z * lightIntensity.z * misWeight * lightDirectionAtten;
    if(bounce > 0)
      currAccumColor.w *= prevPdfA;
    
    *accumColor = currAccumColor;
    *rayFlags   = currRayFlags | (RAY_FLAG_IS_DEAD | RAY_FLAG_HIT_LIGHT);
    return;
  }
  
  const float4 uv         = rndFloat4_Pseudo(a_gen);
  const BsdfSample matSam = MaterialSampleAndEval(matId, lambda, uv, (-1.0f)*ray_dir, hit.norm, hit.uv, misPrev, currRayFlags);
  const float3 bxdfVal    = matSam.val * (1.0f / std::max(matSam.pdf, 1e-20f));
  const float  cosTheta   = std::abs(dot(matSam.dir, hit.norm)); 

  MisData nextBounceData      = *misPrev;        // remember current pdfW for next bounce
  nextBounceData.matSamplePdf = (matSam.flags & RAY_EVENT_S) != 0 ? -1.0f : matSam.pdf; 
  nextBounceData.cosTheta     = cosTheta;   
  *misPrev                    = nextBounceData;

  if(m_intergatorType == INTEGRATOR_STUPID_PT)
  {
    *accumThoroughput *= cosTheta * to_float4(bxdfVal, 0.0f); 
  }
  else if(m_intergatorType == INTEGRATOR_SHADOW_PT || m_intergatorType == INTEGRATOR_MIS_PT)
  {
    const float4 currThoroughput = *accumThoroughput;
    const float4 shadeColor      = *in_shadeColor;
    float4 currAccumColor        = *accumColor;

    currAccumColor.x += currThoroughput.x * shadeColor.x;
    currAccumColor.y += currThoroughput.y * shadeColor.y;
    currAccumColor.z += currThoroughput.z * shadeColor.z;
    if(bounce > 0)
      currAccumColor.w *= prevPdfA;

    *accumColor       = currAccumColor;
    *accumThoroughput = currThoroughput*cosTheta*to_float4(bxdfVal, 0.0f); 
  }

  *rayPosAndNear = to_float4(OffsRayPos(hit.pos, hit.norm, matSam.dir), 0.0f); // todo: use flatNormal for offset
  *rayDirAndFar  = to_float4(matSam.dir, FLT_MAX);
  *rayFlags      = currRayFlags | matSam.flags;
}

void Integrator::kernel_HitEnvironment(uint tid, const uint* rayFlags, const float4* rayDirAndFar, const MisData* a_prevMisData, const float4* accumThoroughput,
                                       float4* accumColor)
{
  const uint currRayFlags = *rayFlags;
  if(!isOutOfScene(currRayFlags))
    return;
  
  const float4 envData  = GetEnvironmentColorAndPdf(to_float3(*rayDirAndFar));
  const float3 envColor = to_float3(envData)/envData.w;                         // explicitly account for pdf; when MIS will be enabled, need to deal with MIS weight also!

  if(m_intergatorType == INTEGRATOR_STUPID_PT)                                  // todo: when explicit sampling will be added, disable contribution here for 'INTEGRATOR_SHADOW_PT'
    *accumColor = (*accumThoroughput) * to_float4(envColor,0);
  else
    *accumColor += (*accumThoroughput) * to_float4(envColor,0);
}


void Integrator::kernel_ContributeToImage(uint tid, const float4* a_accumColor, const RandomGen* gen, const uint* in_pakedXY,
                                          const float3* wavelengths, float4* out_color)
{
  const uint XY = in_pakedXY[tid];
  const uint x  = (XY & 0x0000FFFF);
  const uint y  = (XY & 0xFFFF0000) >> 16;

  float3 color = to_float3(*a_accumColor);

  if(m_spectral_mode) // TODO: spectral framebuffer
  {
    color = SpectrumToXYZ(color, *wavelengths);
    color = XYZToRGB(color);
  }

  float4 colorRes = to_float4(color, 1.0f);
  //if(x == 511 && (y == 1024-340-1))
  //  color = float4(0,0,1,0);
  //if(!std::isfinite(color.x) || !std::isfinite(color.y) || !std::isfinite(color.z))
  //{
  //  int a = 2;
  //  std::cout << "(x,y) = " << x << ", " << y << std::endl; 
  //}
 
  out_color[y*m_winWidth+x] += colorRes;
  m_randomGens[tid] = *gen;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Integrator::NaivePathTrace(uint tid, float4* out_color)
{
  float4 accumColor, accumThroughput;
  float4 rayPosAndNear, rayDirAndFar;
  float3 wavelengths;
  RandomGen gen; 
  MisData   mis;
  uint      rayFlags;
  kernel_InitEyeRay2(tid, m_packedXY.data(), &rayPosAndNear, &rayDirAndFar, &wavelengths, &accumColor, &accumThroughput, &gen, &rayFlags, &mis);

  for(uint depth = 0; depth < m_traceDepth + 1; ++depth) // + 1 due to NaivePT uses additional bounce to hit light 
  {
    float4 shadeColor, hitPart1, hitPart2;
    uint instId = 0;
    kernel_RayTrace2(tid, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &instId, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
    
    kernel_NextBounce(tid, depth, &hitPart1, &hitPart2, &instId, &shadeColor,
                      &rayPosAndNear, &rayDirAndFar, &wavelengths, &accumColor, &accumThroughput, &gen, &mis, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
  }

  kernel_HitEnvironment(tid, &rayFlags, &rayDirAndFar, &mis, &accumThroughput,
                       &accumColor);

  kernel_ContributeToImage(tid, &accumColor, &gen, m_packedXY.data(), &wavelengths, 
                           out_color);
}

void Integrator::PathTrace(uint tid, float4* out_color)
{
  float4 accumColor, accumThroughput;
  float4 rayPosAndNear, rayDirAndFar;
  float3 wavelengths;
  RandomGen gen; 
  MisData   mis;
  uint      rayFlags;
  kernel_InitEyeRay2(tid, m_packedXY.data(), &rayPosAndNear, &rayDirAndFar, &wavelengths, &accumColor, &accumThroughput, &gen, &rayFlags, &mis);
    
  //std::vector<float3> rayPos;
  //std::vector<float4> rayColor;

  for(uint depth = 0; depth < m_traceDepth; depth++) 
  {
    float4   shadeColor, hitPart1, hitPart2;
    uint instId;
    kernel_RayTrace2(tid, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &instId, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
    
    kernel_SampleLightSource(tid, &rayPosAndNear, &rayDirAndFar, &wavelengths, &hitPart1, &hitPart2, &rayFlags,
                             &gen, &shadeColor);

    kernel_NextBounce(tid, depth, &hitPart1, &hitPart2, &instId, &shadeColor,
                      &rayPosAndNear, &rayDirAndFar, &wavelengths, &accumColor, &accumThroughput, &gen, &mis, &rayFlags);

    //rayPos.push_back(float3(rayPosAndNear.x, rayPosAndNear.y, rayPosAndNear.z));
    //rayColor.push_back(rayPosAndNear);

    if(isDeadRay(rayFlags))
      break;
  }

  kernel_HitEnvironment(tid, &rayFlags, &rayDirAndFar, &mis, &accumThroughput,
                        &accumColor);

  kernel_ContributeToImage(tid, &accumColor, &gen, m_packedXY.data(), &wavelengths, out_color);
  
  // Debug draw ray path
  //kernel_ContributePathRayToImage3(out_color, rayColor, rayPos);
}
////////////////////////////////////////////////////
//// input file: /home/roman/Desktop/hydra3spectr/kernel_slicer/../HydraCore3/integrator_pt2.cpp
////////////////////////////////////////////////////
#include "integrator_pt.h"
#include "include/crandom.h"

#include "include/cmaterial.h"
#include "include/cmat_gltf.h"
#include "include/cmat_conductor.h"
#include "include/cmat_glass.h"
#include "include/cmat_diffuse.h"

#include <chrono>
#include <string>

#include "Image2d.h"
using LiteImage::Image2D;
using LiteImage::Sampler;
using LiteImage::ICombinedImageSampler;
using namespace LiteMath;

LightSample Integrator::LightSampleRev(int a_lightId, float2 rands, float3 illiminationPoint)
{
  const uint gtype = m_lights[a_lightId].geomType;
  switch(gtype)
  {
    case LIGHT_GEOM_DIRECT: return directLightSampleRev(m_lights.data() + a_lightId, rands, illiminationPoint);
    case LIGHT_GEOM_SPHERE: return sphereLightSampleRev(m_lights.data() + a_lightId, rands);
    default:                return areaLightSampleRev  (m_lights.data() + a_lightId, rands);
  };
}

float Integrator::LightPdfSelectRev(int a_lightId) 
{ 
  return 1.0f/float(m_lights.size()); // uniform select
}

//static inline float DistanceSquared(float3 a, float3 b)
//{
//  const float3 diff = b - a;
//  return dot(diff, diff);
//}

float Integrator::LightEvalPDF(int a_lightId, float3 illuminationPoint, float3 ray_dir, const float3 lpos, const float3 lnorm)
{
  const uint gtype    = m_lights[a_lightId].geomType;
  const float hitDist = length(illuminationPoint - lpos);
  
  float cosVal = 1.0f;
  switch(gtype)
  {
    case LIGHT_GEOM_SPHERE:
    {
      const float  lradius = m_lights[a_lightId].size.x;
      const float3 lcenter = to_float3(m_lights[a_lightId].pos);
      //if (DistanceSquared(illuminationPoint, lcenter) - lradius*lradius <= 0.0f)
      //  return 1.0f;
      const float3 dirToV  = normalize(lpos - illuminationPoint);
      cosVal = std::abs(dot(dirToV, lnorm));
    }
    break;

    default:
    cosVal  = std::max(dot(ray_dir, -1.0f*lnorm), 0.0f);
    break;
  };
  
  return PdfAtoW(m_lights[a_lightId].pdfA, hitDist, cosVal);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BsdfSample Integrator::MaterialSampleAndEval(uint a_materialId, float3 wavelengths, float4 rands, float3 v, float3 n, float2 tc, 
                                             MisData* a_misPrev, const uint a_currRayFlags)
{
  // implicit strategy
  const float2 texCoordT = mulRows2x4(m_materials[a_materialId].row0[0], m_materials[a_materialId].row1[0], tc);
  const uint   mtype     = as_uint(m_materials[a_materialId].data[UINT_MTYPE]);

  // TODO: read other parameters from texture

  BsdfSample res;
  {
    res.val   = float3(0,0,0);
    res.pdf   = 1.0f;
    res.dir   = float3(0,1,0);
    res.flags = a_currRayFlags;
  }


  switch(mtype)
  {
    case MAT_TYPE_GLTF:
    {
      const uint   texId     = as_uint(m_materials[a_materialId].data[GLTF_UINT_TEXID0]);
      const float3 texColor  = to_float3(m_textures[texId]->sample(texCoordT));
      const float3 color     = to_float3(m_materials[a_materialId].colors[GLTF_COLOR_BASE])*texColor;
      gltfSampleAndEval(m_materials.data() + a_materialId, rands, v, n, tc, color, &res);
      break;
    }
    case MAT_TYPE_GLASS:
    {
      glassSampleAndEval(m_materials.data() + a_materialId, rands, v, n, tc, &res, a_misPrev);
      break;
    }
    case MAT_TYPE_CONDUCTOR:
    {
      const uint   texId     = as_uint(m_materials[a_materialId].data[CONDUCTOR_TEXID0]);
      const float2 texCoordT = mulRows2x4(m_materials[a_materialId].row0[0], m_materials[a_materialId].row1[0], tc);
      const float3 alphaTex  = to_float3(m_textures[texId]->sample(texCoordT));
      
      const float2 alpha = float2(m_materials[a_materialId].data[CONDUCTOR_ROUGH_V], m_materials[a_materialId].data[CONDUCTOR_ROUGH_U]);
      if(trEffectivelySmooth(alpha))
        conductorSmoothSampleAndEval(m_materials.data() + a_materialId, m_spectra.data(), wavelengths, rands, v, n, tc, &res);
      else
        conductorRoughSampleAndEval(m_materials.data() + a_materialId, m_spectra.data(), wavelengths, rands, v, n, tc, alphaTex, &res);
      
      break;
    }
    case MAT_TYPE_DIFFUSE:
    {
      const uint   texId       = as_uint(m_materials[a_materialId].data[DIFFUSE_TEXID0]);
      // const float3 reflectance = to_float3(m_materials[a_materialId].colors[DIFFUSE_COLOR]); 
      const float3 texColor    = to_float3(m_textures[texId]->sample(texCoordT));
      const float3 color       = texColor;

      diffuseSampleAndEval(m_materials.data() + a_materialId, m_spectra.data(), wavelengths, rands, v, n, tc, color, &res);

      break;
    }
    default:
      break;
  }

  return res;
}

BsdfEval Integrator::MaterialEval(uint a_materialId, float3 wavelengths, float3 l, float3 v, float3 n, float2 tc)
{
  // explicit strategy
  const float2 texCoordT = mulRows2x4(m_materials[a_materialId].row0[0], m_materials[a_materialId].row1[0], tc);
  const uint   mtype     = as_uint(m_materials[a_materialId].data[UINT_MTYPE]);

  // TODO: read other parameters from texture
  BsdfEval res;
  {
    res.val = float3(0,0,0);
    res.pdf   = 0.0f;
  }

  switch(mtype)
  {
    case MAT_TYPE_GLTF:
    {
      const uint   texId     = as_uint(m_materials[a_materialId].data[GLTF_UINT_TEXID0]);
      const float3 texColor  = to_float3(m_textures[texId]->sample(texCoordT));
      const float3 color     = to_float3(m_materials[a_materialId].colors[GLTF_COLOR_BASE])*texColor;
      gltfEval(m_materials.data() + a_materialId, l, v, n, tc, color, &res);
      break;
    }
    case MAT_TYPE_GLASS:
    {
      glassEval(m_materials.data() + a_materialId, l, v, n, tc, {}, &res);
      break;
    }
    case MAT_TYPE_CONDUCTOR: 
    {
      const uint   texId     = as_uint(m_materials[a_materialId].data[CONDUCTOR_TEXID0]);
      const float3 alphaTex  = to_float3(m_textures[texId]->sample(texCoordT));

      const float2 alpha = float2(m_materials[a_materialId].data[CONDUCTOR_ROUGH_V], m_materials[a_materialId].data[CONDUCTOR_ROUGH_U]);
      if(trEffectivelySmooth(alpha))
        conductorSmoothEval(m_materials.data() + a_materialId, wavelengths, l, v, n, tc, &res);
      else
        conductorRoughEval(m_materials.data() + a_materialId, m_spectra.data(), wavelengths, l, v, n, tc, alphaTex, &res);

      break;
    }
    case MAT_TYPE_DIFFUSE:
    {
      const uint   texId       = as_uint(m_materials[a_materialId].data[DIFFUSE_TEXID0]);
      // const float3 reflectance = to_float3(m_materials[a_materialId].colors[DIFFUSE_COLOR]); 
      const float3 texColor    = to_float3(m_textures[texId]->sample(texCoordT));
      const float3 color       = texColor;

      diffuseEval(m_materials.data() + a_materialId, m_spectra.data(), wavelengths, l, v, n, tc, color, &res);

      break;
    }
    case MAT_TYPE_FILM: 
    {
      const uint   texId     = as_uint(m_materials[a_materialId].data[FILM_TEXID0]);
      const float3 alphaTex  = to_float3(m_textures[texId]->sample(texCoordT));

      const float2 alpha = float2(m_materials[a_materialId].data[FILM_ROUGH_V], m_materials[a_materialId].data[FILM_ROUGH_U]);
      if(trEffectivelySmooth(alpha))
        filmSmoothEval(m_materials.data() + a_materialId, wavelengths, l, v, n, tc, &res);
      else
        filmRoughEval(m_materials.data() + a_materialId, m_spectra.data(), wavelengths, l, v, n, tc, alphaTex, &res);

      break;
    }
    default:
      break;
  }

  return res;
}

float4 Integrator::GetEnvironmentColorAndPdf(float3 a_dir)
{
  return m_envColor;
}

uint Integrator::RemapMaterialId(uint a_mId, int a_instId)
{
  const int remapListId  = m_remapInst[a_instId];
  if(remapListId == -1)
    return a_mId;

  const int r_offset     = m_allRemapListsOffsets[remapListId];
  const int r_size       = m_allRemapListsOffsets[remapListId+1] - r_offset;
  const int2 offsAndSize = int2(r_offset, r_size);
  
  uint res = a_mId;
  
  // for (int i = 0; i < offsAndSize.y; i++) // linear search version
  // {
  //   int idRemapFrom = m_allRemapLists[offsAndSize.x + i * 2 + 0];
  //   int idRemapTo   = m_allRemapLists[offsAndSize.x + i * 2 + 1];
  //   if (idRemapFrom == a_mId) {
  //     res = idRemapTo;
  //     break;
  //   }
  // }

  int low  = 0;
  int high = offsAndSize.y - 1;              // binary search version
  
  while (low <= high)
  {
    const int mid         = low + ((high - low) / 2);
    const int idRemapFrom = m_allRemapLists[offsAndSize.x + mid * 2 + 0];
    if (uint(idRemapFrom) >= a_mId)
      high = mid - 1;
    else //if(a[mid]<i)
      low = mid + 1;
  }

  if (high+1 < offsAndSize.y)
  {
    const int idRemapFrom = m_allRemapLists[offsAndSize.x + (high + 1) * 2 + 0];
    const int idRemapTo   = m_allRemapLists[offsAndSize.x + (high + 1) * 2 + 1];
    res                   = (uint(idRemapFrom) == a_mId) ? uint(idRemapTo) : a_mId;
  }

  return res;
} 

void Integrator::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  if(std::string(a_funcName) == "NaivePathTrace" || std::string(a_funcName) == "NaivePathTraceBlock")
    a_out[0] = naivePtTime;
  else if(std::string(a_funcName) == "PathTrace" || std::string(a_funcName) == "PathTraceBlock")
    a_out[0] = shadowPtTime;
  else if(std::string(a_funcName) == "RayTrace" || std::string(a_funcName) == "RayTraceBlock")
    a_out[0] = raytraceTime;
}
////////////////////////////////////////////////////
//// input file: /home/roman/Desktop/hydra3spectr/kernel_slicer/../HydraCore3/integrator_rt.cpp
////////////////////////////////////////////////////
#include "integrator_pt.h"
#include "include/crandom.h"

#include <chrono>
#include <string>

#include "Image2d.h"
using LiteImage::Image2D;
using LiteImage::Sampler;
using LiteImage::ICombinedImageSampler;
using namespace LiteMath;

void Integrator::kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  const uint inBlockIdX = tidX % 8; // 8x8 blocks
  const uint inBlockIdY = tidY % 8; // 8x8 blocks
 
  const uint localIndex = inBlockIdY*8 + inBlockIdX;
  const uint wBlocks    = m_winWidth/8;

  const uint blockX     = tidX/8;
  const uint blockY     = tidY/8;
  const uint offset     = (blockX + blockY*wBlocks)*8*8 + localIndex;

  out_pakedXY[offset] = ((tidY << 16) & 0xFFFF0000) | (tidX & 0x0000FFFF);
}

void Integrator::kernel_InitEyeRay(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar) // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!
{
  const uint XY = packedXY[tid];

  const uint x = (XY & 0x0000FFFF);
  const uint y = (XY & 0xFFFF0000) >> 16;

  float3 rayDir = EyeRayDirNormalized((float(x)+0.5f)/float(m_winWidth), (float(y)+0.5f)/float(m_winHeight), m_projInv);
  float3 rayPos = float3(0,0,0);

  transform_ray3f(m_worldViewInv, 
                  &rayPos, &rayDir);
  
  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, FLT_MAX);
}

void Integrator::kernel_InitEyeRay3(uint tid, const uint* packedXY, 
                                   float4* rayPosAndNear, float4* rayDirAndFar,
                                   float4* accumColor,    float4* accumuThoroughput,
                                   uint* rayFlags) // 
{
  *accumColor        = make_float4(0,0,0,1);
  *accumuThoroughput = make_float4(1,1,1,1);
  //RandomGen genLocal = m_randomGens[tid];
  *rayFlags          = 0;

  const uint XY = packedXY[tid];

  const uint x = (XY & 0x0000FFFF);
  const uint y = (XY & 0xFFFF0000) >> 16;

  float3 rayDir = EyeRayDirNormalized((float(x))/float(m_winWidth), 
                                      (float(y))/float(m_winHeight), m_projInv);
  float3 rayPos = float3(0,0,0);

  transform_ray3f(m_worldViewInv, &rayPos, &rayDir);
  
  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, FLT_MAX);
}


bool Integrator::kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar,
                                 Lite_Hit* out_hit, float2* out_bars)
{
  const float4 rayPos = *rayPosAndNear;
  const float4 rayDir = *rayDirAndFar ;

  CRT_Hit hit = m_pAccelStruct->RayQuery_NearestHit(rayPos, rayDir);
  
  Lite_Hit res;
  res.primId = hit.primId;
  res.instId = hit.instId;
  res.geomId = hit.geomId;
  res.t      = hit.t;

  float2 baricentrics = float2(hit.coords[0], hit.coords[1]);
 
  *out_hit  = res;
  *out_bars = baricentrics;
  return (res.primId != -1);
}


void Integrator::kernel_RealColorToUint32(uint tid, float4* a_accumColor, uint* out_color)
{
  out_color[tid] = RealColorToUint32(*a_accumColor);
}

void Integrator::kernel_GetRayColor(uint tid, const Lite_Hit* in_hit, const uint* in_pakedXY, 
  uint* out_color)
{ 
  const Lite_Hit lhit = *in_hit;
  if(lhit.geomId == -1)
  {
    out_color[tid] = 0;
    return;
  }

  const uint32_t matId = m_matIdByPrimId[m_matIdOffsets[lhit.geomId] + lhit.primId];
  const float4 mdata   = m_materials[matId].colors[GLTF_COLOR_BASE];
  const float3 color   = mdata.w > 0.0f ? clamp(float3(mdata.w,mdata.w,mdata.w), 0.0f, 1.0f) : to_float3(mdata);

  const uint XY = in_pakedXY[tid];
  const uint x  = (XY & 0x0000FFFF);
  const uint y  = (XY & 0xFFFF0000) >> 16;

  out_color[y*m_winWidth+x] = RealColorToUint32_f3(color); 
}


float3 Integrator::MaterialEvalWhitted(uint a_materialId, float3 l, float3 v, float3 n, float2 tc)
{
  const uint   texId     = as_uint(m_materials[a_materialId].data[GLTF_UINT_TEXID0]);
  const float2 texCoordT = mulRows2x4(m_materials[a_materialId].row0[0], m_materials[a_materialId].row1[0], tc);
  const float3 texColor  = to_float3(m_textures[texId]->sample(texCoordT));
  const float3 color     = to_float3(m_materials[a_materialId].colors[GLTF_COLOR_BASE])*texColor;
  return lambertEvalBSDF(l, v, n)*color;
}

BsdfSample Integrator::MaterialSampleWhitted(uint a_materialId, float3 v, float3 n, float2 tc)
{ 
  const uint  type       = as_uint(m_materials[a_materialId].data[UINT_MTYPE]);
  const float3 specular  = to_float3(m_materials[a_materialId].colors[GLTF_COLOR_METAL]);
  const float3 coat      = to_float3(m_materials[a_materialId].colors[GLTF_COLOR_COAT]);
  const float  roughness = 1.0f - m_materials[a_materialId].data[GLTF_FLOAT_GLOSINESS];
  float alpha            = m_materials[a_materialId].data[GLTF_FLOAT_ALPHA];
  
  const float3 pefReflDir = reflect((-1.0f)*v, n);
  const float3 reflColor  = alpha*specular + (1.0f - alpha)*coat;

  //if(a_materialId == 4)
  //{
  //  int a = 2;
  //}

  BsdfSample res;
  res.dir   = pefReflDir;
  res.val   = reflColor;
  res.pdf   = 1.0f;
  res.flags = RAY_EVENT_S;
  return res;
}


void Integrator::kernel_RayBounce(uint tid, uint bounce, const float4* in_hitPart1, const float4* in_hitPart2,
                                  float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumThoroughput,
                                  uint* rayFlags)
{
  const uint currRayFlags = *rayFlags;
  if(isDeadRay(currRayFlags))
    return;

  const uint32_t matId = extractMatId(currRayFlags);

  // process surface hit case
  //
  const float3 ray_dir = to_float3(*rayDirAndFar);
  //const float3 ray_pos = to_float3(*rayPosAndNear);

  const float4 data1 = *in_hitPart1;
  const float4 data2 = *in_hitPart2;

  SurfaceHit hit;
  hit.pos  = to_float3(data1);
  hit.norm = to_float3(data2);
  hit.uv   = float2(data1.w, data2.w);

  // process light hit case
  //
  if(as_uint(m_materials[matId].data[UINT_MTYPE]) == MAT_TYPE_LIGHT_SOURCE)
  {
    const uint   texId          = as_uint(m_materials[matId].data[GLTF_UINT_TEXID0]);
    const float2 texCoordT      = mulRows2x4(m_materials[matId].row0[0], m_materials[matId].row1[0], hit.uv);
    const float3 texColor       = to_float3(m_textures[texId]->sample(texCoordT));

    const float3 lightIntensity = to_float3(m_materials[matId].colors[GLTF_COLOR_BASE])*texColor;
    const uint lightId          = as_uint(m_materials[matId].data[UINT_LIGHTID]);
    float lightDirectionAtten   = (lightId == 0xFFFFFFFF) ? 1.0f : dot(to_float3(*rayDirAndFar), float3(0,-1,0)) < 0.0f ? 1.0f : 0.0f; // TODO: read light info, gety light direction and e.t.c;


    float4 currAccumColor      = *accumColor;
    float4 currAccumThroughput = *accumThoroughput;

    currAccumColor.x += currAccumThroughput.x * lightIntensity.x * lightDirectionAtten;
    currAccumColor.y += currAccumThroughput.y * lightIntensity.y * lightDirectionAtten;
    currAccumColor.z += currAccumThroughput.z * lightIntensity.z * lightDirectionAtten;

    *accumColor = currAccumColor;
    *rayFlags   = currRayFlags | (RAY_FLAG_IS_DEAD | RAY_FLAG_HIT_LIGHT);
    return;
  }

  float4 shadeColor = float4(0.0f, 0.0f, 0.0f, 1.0f);
  for(uint lightId = 0; lightId < m_lights.size(); ++lightId)
  {
    const float3 lightPos = to_float3(m_lights[lightId].pos);
    const float hitDist   = sqrt(dot(hit.pos - lightPos, hit.pos - lightPos));

    const float3 shadowRayDir = normalize(lightPos - hit.pos);
    const float3 shadowRayPos = hit.pos + hit.norm * std::max(maxcomp(hit.pos), 1.0f) * 5e-6f; // TODO: see Ray Tracing Gems, also use flatNormal for offset

    const bool inShadow = m_pAccelStruct->RayQuery_AnyHit(to_float4(shadowRayPos, 0.0f), to_float4(shadowRayDir, hitDist * 0.9995f));

    if(!inShadow && dot(shadowRayDir, to_float3(m_lights[lightId].norm)) < 0.0f)
    {
      const float3 matSamColor = MaterialEvalWhitted(matId, shadowRayDir, (-1.0f)*ray_dir, hit.norm, hit.uv);
      const float cosThetaOut  = std::max(dot(shadowRayDir, hit.norm), 0.0f);
      shadeColor += to_float4(to_float3(m_lights[lightId].intensity) * matSamColor*cosThetaOut / (hitDist * hitDist), 0.0f);
    }
  }

  const BsdfSample matSam = MaterialSampleWhitted(matId, (-1.0f)*ray_dir, hit.norm, hit.uv);
  const float3 bxdfVal    = matSam.val;
  const float  cosTheta   = dot(matSam.dir, hit.norm);

  const float4 currThoroughput = *accumThoroughput;
  float4 currAccumColor        = *accumColor;

  currAccumColor.x += currThoroughput.x * shadeColor.x;
  currAccumColor.y += currThoroughput.y * shadeColor.y;
  currAccumColor.z += currThoroughput.z * shadeColor.z;

  *accumColor       = currAccumColor;
  *accumThoroughput = currThoroughput * cosTheta * to_float4(bxdfVal, 0.0f);

  *rayPosAndNear = to_float4(OffsRayPos(hit.pos, hit.norm, matSam.dir), 0.0f);
  *rayDirAndFar  = to_float4(matSam.dir, FLT_MAX);
  *rayFlags      = currRayFlags | matSam.flags;
}

void Integrator::kernel_ContributeToImage3(uint tid, const float4* a_accumColor, const uint* in_pakedXY, float4* out_color)
{
  const uint XY = in_pakedXY[tid];
  const uint x  = (XY & 0x0000FFFF);
  const uint y  = (XY & 0xFFFF0000) >> 16;

  float4 color = *a_accumColor;
  out_color[y*m_winWidth+x] += color;
}

static inline float2 clipSpaceToScreenSpace(float4 a_pos, const float fw, const float fh)
{
  const float x = a_pos.x * 0.5f + 0.5f;
  const float y = a_pos.y * 0.5f + 0.5f;
  return make_float2(x * fw, y * fh);
}

static inline float4x4 make_float4x4(const float* a_data)
{
  float4x4 matrix;
  matrix.m_col[0] = make_float4(a_data[0], a_data[1], a_data[2], a_data[3]);
  matrix.m_col[1] = make_float4(a_data[4], a_data[5], a_data[6], a_data[7]);
  matrix.m_col[2] = make_float4(a_data[8], a_data[9], a_data[10], a_data[11]);
  matrix.m_col[3] = make_float4(a_data[12], a_data[13], a_data[14], a_data[15]);
  return matrix;
}

static inline float2 worldPosToScreenSpace(float3 a_wpos, const int width, const int height, 
  float4x4 worldView, float4x4 proj)
{
  const float4 posWorldSpace  = to_float4(a_wpos, 1.0f);
  const float4 posCamSpace    = mul4x4x4(worldView, posWorldSpace);
  const float4 posNDC         = mul4x4x4(proj, posCamSpace);
  const float4 posClipSpace   = posNDC * (1.0f / fmax(posNDC.w, DEPSILON));
  const float2 posScreenSpace = clipSpaceToScreenSpace(posClipSpace, width, height);
  return posScreenSpace;
}

void drawLine(const float3 a_pos1, const float3 a_pos2, float4 * a_outColor, const int a_winWidth,
  const int a_winHeight, const float4 a_rayColor1, const float4 a_rayColor2, const bool a_blendColor,
  const bool a_multDepth, const int a_spp)
{
  const int dx   = abs(a_pos2.x - a_pos1.x);
  const int dy   = abs(a_pos2.y - a_pos1.y);

  const int step = dx > dy ? dx : dy;

  float x_inc    = dx / (float)step;
  float y_inc    = dy / (float)step;

  if (a_pos1.x > a_pos2.x) x_inc *= -1.0f;
  if (a_pos1.y > a_pos2.y) y_inc *= -1.0f;

  float x = a_pos1.x;
  float y = a_pos1.y;

  const float depthMult1 = tanh(a_pos1.z * 0.25f) * 0.5f + 0.5f; // rescale for 0 - 1
  const float depthMult2 = tanh(a_pos2.z * 0.25f) * 0.5f + 0.5f; // rescale for 0 - 1

  for (int i = 0; i <= step; ++i) 
  {
    if (x >= 0 && x <= a_winWidth - 1 && y >= 0 && y <= a_winHeight - 1)
    {
      float4 color;
      float weight    = (float)(i) / (float)(step);
      
      float depthMult = 1.0f; 
      
      if (a_multDepth) 
        depthMult = lerp(depthMult1, depthMult2, weight);
      
      if (!a_blendColor)
        weight = 0.0f;

      color = lerp(a_rayColor1, a_rayColor2, weight) * depthMult;
         
      a_outColor[(int)(y)*a_winWidth + (int)(x)] += color * a_spp;
    }
 
    x += x_inc;
    y += y_inc;
  }
  if (a_pos1.x >= 0 && a_pos1.x <= a_winWidth - 1 && a_pos1.y >= 0 && a_pos1.y <= a_winHeight - 1)
    a_outColor[(int)(a_pos1.y)*a_winWidth + (int)(a_pos1.x)] = float4(0, a_spp, 0, 0);
}

void Integrator::kernel_ContributePathRayToImage3(float4* out_color, 
  const std::vector<float4>& a_rayColor, std::vector<float3>& a_rayPos)
{  
  for (int i = 1; i < a_rayPos.size(); ++i)
  {
    const float2 posScreen1 = worldPosToScreenSpace(a_rayPos[i - 1], m_winWidth, m_winHeight, m_worldView, m_proj);
    const float2 posScreen2 = worldPosToScreenSpace(a_rayPos[i - 0], m_winWidth, m_winHeight, m_worldView, m_proj);
    
    const float3 pos1 = float3(posScreen1.x, posScreen1.y, a_rayPos[i - 1].z);
    const float3 pos2 = float3(posScreen2.x, posScreen2.y, a_rayPos[i    ].z);

    // fix color
    //const float4 rayColor = float4(1, 1, 1, 1); 

    // shade color
    //const float4 rayColor1 = a_rayColor[i - 1]; 
    //const float4 rayColor2 = a_rayColor[i - 0];

    // direction color
    //const float4 rayColor1 = (a_rayColor[i - 1]) * 0.5f + 0.5f;
    //const float4 rayColor2 = (a_rayColor[i - 0]) * 0.5f + 0.5f;

    // position color with rescale to 0-1
    const float scaleSize  = 0.5f;
    const float4 rayColor1 = float4(tanh(a_rayPos[i - 1].x * scaleSize), tanh(a_rayPos[i - 1].y * scaleSize), tanh(a_rayPos[i - 1].z * scaleSize), 0) * 0.5f + 0.5f;
    const float4 rayColor2 = float4(tanh(a_rayPos[i - 0].x * scaleSize), tanh(a_rayPos[i - 0].y * scaleSize), tanh(a_rayPos[i - 0].z * scaleSize), 0) * 0.5f + 0.5f;
        
    drawLine(pos1, pos2, out_color, m_winWidth, m_winHeight, rayColor1, rayColor2, true, true, m_spp);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Integrator::PackXY(uint tidX, uint tidY)
{
  kernel_PackXY(tidX, tidY, m_packedXY.data());
}

void Integrator::CastSingleRay(uint tid, uint* out_color)
{
  float4 rayPosAndNear, rayDirAndFar;
  kernel_InitEyeRay(tid, m_packedXY.data(), &rayPosAndNear, &rayDirAndFar);

  Lite_Hit hit; 
  float2   baricentrics; 
  if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &hit, &baricentrics))
    return;
  
  kernel_GetRayColor(tid, &hit, m_packedXY.data(), out_color);
}

void Integrator::RayTrace(uint tid, float4* out_color)
{
  float4 accumColor, accumThroughput;
  float4 rayPosAndNear, rayDirAndFar;
  uint      rayFlags = 0;
  kernel_InitEyeRay3(tid, m_packedXY.data(), 
                     &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThroughput, &rayFlags);

  for(uint depth = 0; depth < m_traceDepth; depth++)
  {
    float4 hitPart1, hitPart2;
    uint instId;
    kernel_RayTrace2(tid, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &instId, &rayFlags);
    if(isDeadRay(rayFlags))
      break;

    kernel_RayBounce(tid, depth, &hitPart1, &hitPart2,
                     &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThroughput, &rayFlags);

    if(isDeadRay(rayFlags))
      break;
  }

//  kernel_HitEnvironment(tid, &rayFlags, &rayDirAndFar, &mis, &accumThroughput,
//                        &accumColor);

  kernel_ContributeToImage3(tid, &accumColor, m_packedXY.data(), out_color);
}
