#pragma once
#include "cglobals.h"
#include "crandom.h"
#include "cmaterial.h"

static inline void gltfSampleAndEval(const GLTFMaterial* a_materials, int a_materialId, float4 rands, float3 v, float3 n, float2 tc, 
                                     const float3 color,
                                     BsdfSample* pRes)
{
  const uint   cflags     = a_materials[a_materialId].cflags;
  const float3 specular   = to_float3(a_materials[a_materialId].metalColor);
  const float3 coat       = to_float3(a_materials[a_materialId].coatColor);
  const float  roughness  = 1.0f - a_materials[a_materialId].glosiness;
  float        alpha      = a_materials[a_materialId].alpha;
  const float  fresnelIOR = a_materials[a_materialId].ior;

  if(cflags == GLTF_COMPONENT_METAL) // assume only GGX-based metal component set
    alpha = 1.0f;

  float3 ggxDir;
  float  ggxPdf; 
  float  ggxVal;

  if(roughness == 0.0f) // perfect specular reflection in coating or metal layer
  {
    const float3 pefReflDir = reflect((-1.0f)*v, n);
    const float cosThetaOut = dot(pefReflDir, n);
    ggxDir = pefReflDir;
    ggxVal = (cosThetaOut <= 1e-6f) ? 0.0f : (1.0f/std::max(cosThetaOut, 1e-6f));  // BSDF is multiplied (outside) by cosThetaOut. For mirrors this shouldn't be done, so we pre-divide here instead.
    ggxPdf = 1.0f;
  }
  else
  {
    ggxDir = ggxSample(float2(rands.x, rands.y), v, n, roughness);
    ggxPdf = ggxEvalPDF (ggxDir, v, n, roughness); 
    ggxVal = ggxEvalBSDF(ggxDir, v, n, roughness);
  }

  const float3 lambertDir = lambertSample(float2(rands.x, rands.y), v, n);
  const float  lambertPdf = lambertEvalPDF(lambertDir, v, n);
  const float  lambertVal = lambertEvalBSDF(lambertDir, v, n);

  // (1) select between metal and dielectric via rands.z
  //
  float pdfSelect = 1.0f;
  if(rands.z < alpha) // select metall
  {
    pdfSelect *= alpha;
    const float  VdotH = dot(v,normalize(v + ggxDir));
    pRes->direction = ggxDir;
    pRes->color     = ggxVal*alpha*hydraFresnelCond(specular, VdotH, fresnelIOR, roughness); //TODO: disable fresnel here for mirrors
    pRes->pdf       = ggxPdf;
    pRes->flags     = (roughness == 0.0f) ? RAY_EVENT_S : RAY_FLAG_HAS_NON_SPEC;
  }
  else                // select dielectric
  {
    pdfSelect *= 1.0f - alpha;
    
    // (2) now select between specular and diffise via rands.w
    //
    const float f_i = FrDielectricPBRT(std::abs(dot(v,n)), 1.0f, fresnelIOR); 
    const float m_specular_sampling_weight = a_materials[a_materialId].data[MI_SSW];
    
    float prob_specular = f_i * m_specular_sampling_weight;
    float prob_diffuse  = (1.f - f_i) * (1.f - m_specular_sampling_weight);
    if(prob_diffuse != 0.0f && prob_diffuse != 0.0f)
    {
      prob_specular = prob_specular / (prob_specular + prob_diffuse);
      prob_diffuse  = 1.f - prob_specular;
    }
    else
    {
      prob_diffuse  = 1.0f;
      prob_specular = 0.0f;
    }
    float choicePdf = ((cflags & GLTF_COMPONENT_COAT) == 0) ? 0.0f : prob_specular; // if don't have coal layer, never select it
    if(rands.w < prob_specular) // specular
    {
      pdfSelect *= choicePdf;
      pRes->direction = ggxDir;
      pRes->color     = ggxVal*coat*(1.0f - alpha)*f_i;
      pRes->pdf       = ggxPdf;
      pRes->flags     = (roughness == 0.0f) ? RAY_EVENT_S : RAY_FLAG_HAS_NON_SPEC;
    } 
    else
    {
      pdfSelect *= (1.0f-choicePdf); // lambert
      pRes->direction = lambertDir;
      pRes->color     = lambertVal*color*(1.0f - alpha);
      pRes->pdf       = lambertPdf;
      pRes->flags     = RAY_FLAG_HAS_NON_SPEC;
      
      if((cflags & GLTF_COMPONENT_COAT) != 0 && (cflags & GLTF_COMPONENT_LAMBERT) != 0) // Plastic, account for retroreflection between surface and coating layer
      {
        const float m_fdr_int = a_materials[a_materialId].data[MI_FDR_INT];
        const float f_o = FrDielectricPBRT(std::abs(dot(lambertDir,n)), 1.0f, fresnelIOR); 
        pRes->color *= (1.f - f_i) * (1.f - f_o) / (fresnelIOR*fresnelIOR*(1.f - m_fdr_int));
      }
    }
  }   
  pRes->pdf *= pdfSelect;
}