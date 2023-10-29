/////////////////////////////////////////////////////////////////////
/////////////  Required  Shader Features ////////////////////////////
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/////////////////// include files ///////////////////////////////////
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/////////////////// declarations in class ///////////////////////////
/////////////////////////////////////////////////////////////////////
#ifndef uint32_t
#define uint32_t uint
#endif
#define FLT_MAX 1e37f
#define FLT_MIN -1e37f
#define FLT_EPSILON 1e-6f
#define DEG_TO_RAD  0.017453293f
#define unmasked
#define half  float16_t
#define half2 f16vec2
#define half3 f16vec3
#define half4 f16vec4
bool  isfinite(float x)            { return !isinf(x); }
float copysign(float mag, float s) { return abs(mag)*sign(s); }

struct complex
{
  float re, im;
};

complex make_complex(float re, float im) { 
  complex res;
  res.re = re;
  res.im = im;
  return res;
}

complex to_complex(float re)              { return make_complex(re, 0.0f);}
complex complex_add(complex a, complex b) { return make_complex(a.re + b.re, a.im + b.im); }
complex complex_sub(complex a, complex b) { return make_complex(a.re - b.re, a.im - b.im); }
complex complex_mul(complex a, complex b) { return make_complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
complex complex_div(complex a, complex b) {
  const float scale = 1 / (b.re * b.re + b.im * b.im);
  return make_complex(scale * (a.re * b.re + a.im * b.im), scale * (a.im * b.re - a.re * b.im));
}

complex real_add_complex(float value, complex z) { return complex_add(to_complex(value),z); }
complex real_sub_complex(float value, complex z) { return complex_sub(to_complex(value),z); }
complex real_mul_complex(float value, complex z) { return complex_mul(to_complex(value),z); }
complex real_div_complex(float value, complex z) { return complex_div(to_complex(value),z); }

complex complex_add_real(complex z, float value) { return complex_add(z, to_complex(value)); }
complex complex_sub_real(complex z, float value) { return complex_sub(z, to_complex(value)); }
complex complex_mul_real(complex z, float value) { return complex_mul(z, to_complex(value)); }
complex complex_div_real(complex z, float value) { return complex_div(z, to_complex(value)); }

float real(complex z) { return z.re;}
float imag(complex z) { return z.im; }
float complex_norm(complex z) { return z.re * z.re + z.im * z.im; }
float complex_abs(complex z) { return sqrt(complex_norm(z)); }
complex complex_sqrt(complex z) 
{
  float n = complex_abs(z);
  float t1 = sqrt(0.5f * (n + abs(z.re)));
  float t2 = 0.5f * z.im / t1;
  if (n == 0.0f)
    return to_complex(0.0f);
  if (z.re >= 0.0f)
    return make_complex(t1, t2);
  else
    return make_complex(abs(t2), copysign(t1, z.im));
}

const uint RAY_FLAG_IS_DEAD = 0x80000000;
const uint RAY_FLAG_OUT_OF_SCENE = 0x40000000;
const uint RAY_FLAG_HIT_LIGHT = 0x20000000;
const uint RAY_FLAG_HAS_NON_SPEC = 0x10000000;
const uint RAY_FLAG_HAS_INV_NORMAL = 0x08000000;
struct Lite_HitT
{
  float t;
  int   primId; 
  int   instId;
  int   geomId;
};
#define Lite_Hit Lite_HitT
struct SurfaceHitT
{
  vec3 pos;
  vec3 norm;
  vec2 uv;
};
#define SurfaceHit SurfaceHitT
const float GEPSILON = 1e-5f;
const float DEPSILON = 1e-20f;
struct MisData
{
  float matSamplePdf; ///< previous angle pdf (pdfW) that were used for sampling material. if < 0, then material sample was pure specular 
  float cosTheta;     ///< previous dot(matSam.dir, hit.norm)
  float ior;          ///< previous ior
  float dummy;        ///< dummy for 4 float
};
struct RandomGenT
{
  uvec2 state;

};
#define RandomGen RandomGenT
const uint LIGHT_GEOM_RECT = 1;
const uint LIGHT_GEOM_DISC = 2;
const uint LIGHT_GEOM_SPHERE = 3;
const uint LIGHT_GEOM_DIRECT = 4;
struct LightSource
{
  mat4 matrix;    ///<! translation in matrix is always (0,0,0,1)
  vec4   pos;       ///<! translation aclually stored here
  vec4   intensity; ///<! brightress, i.e. screen value if light is visable directly
  vec4   norm;
  vec2   size;
  float    pdfA;
  uint     geomType;  ///<! LIGHT_GEOM_RECT, LIGHT_GEOM_DISC, LIGHT_GEOM_SPHERE
  vec4   ids;       /// (spec_id, tex_id, ies_id, unused)
};
struct LightSample
{
  vec3 pos;
  vec3 norm;
};
struct BsdfSample
{
  vec3 val;
  vec3 dir;
  float  pdf; 
  uint   flags;
  float  ior;
};
struct BsdfEval
{
  vec3 val;
  float  pdf; 
};
const uint GLTF_COMPONENT_LAMBERT = 1;
const uint GLTF_COMPONENT_COAT = 2;
const uint GLTF_COMPONENT_METAL = 4;
const uint GLTF_METAL_PERF_MIRROR = 8;
const uint GLTF_COMPONENT_ORENNAYAR = 16;
const uint MAT_TYPE_GLTF = 1;
const uint MAT_TYPE_GLASS = 2;
const uint MAT_TYPE_CONDUCTOR = 3;
const uint MAT_TYPE_DIFFUSE = 4;
const uint MAT_TYPE_LIGHT_SOURCE = 0xEFFFFFFF;
const uint RAY_EVENT_S = 1;
const uint RAY_EVENT_D = 2;
const uint RAY_EVENT_G = 4;
const uint RAY_EVENT_T = 8;
const uint RAY_EVENT_V = 16;
const uint RAY_EVENT_TOUT = 32;
const uint RAY_EVENT_TNINGLASS = 64;
const uint UINT_MTYPE = 0;
const uint UINT_CFLAGS = 1;
const uint UINT_LIGHTID = 2;
const uint UINT_MAIN_LAST_IND = 3;
const uint GLTF_COLOR_BASE = 0;
const uint GLTF_COLOR_COAT = 1;
const uint GLTF_COLOR_METAL = 2;
const uint GLTF_COLOR_LAST_IND = GLTF_COLOR_METAL;
const uint GLTF_FLOAT_MI_FDR_INT = UINT_MAIN_LAST_IND + 0;
const uint GLTF_FLOAT_MI_FDR_EXT = UINT_MAIN_LAST_IND + 1;
const uint GLTF_FLOAT_MI_SSW = UINT_MAIN_LAST_IND + 2;
const uint GLTF_FLOAT_ALPHA = UINT_MAIN_LAST_IND + 3;
const uint GLTF_FLOAT_GLOSINESS = UINT_MAIN_LAST_IND + 4;
const uint GLTF_FLOAT_IOR = UINT_MAIN_LAST_IND + 5;
const uint GLTF_FLOAT_ROUGH_ORENNAYAR = UINT_MAIN_LAST_IND + 6;
const uint GLTF_UINT_TEXID0 = UINT_MAIN_LAST_IND + 7;
const uint GLTF_CUSTOM_LAST_IND = GLTF_UINT_TEXID0;
const uint GLASS_COLOR_REFLECT = 0;
const uint GLASS_COLOR_TRANSP = 1;
const uint GLASS_COLOR_LAST_IND = GLASS_COLOR_TRANSP;
const uint GLASS_FLOAT_GLOSS_REFLECT = UINT_MAIN_LAST_IND + 0;
const uint GLASS_FLOAT_GLOSS_TRANSP = UINT_MAIN_LAST_IND + 1;
const uint GLASS_FLOAT_IOR = UINT_MAIN_LAST_IND + 2;
const uint GLASS_CUSTOM_LAST_IND = GLASS_FLOAT_IOR;
const uint EMISSION_COLOR = 0;
const uint EMISSION_COLOR_LAST_IND = EMISSION_COLOR;
const uint EMISSION_TEXID0 = UINT_MAIN_LAST_IND + 0;
const uint EMISSION_SPECID0 = UINT_MAIN_LAST_IND + 1;
const uint EMISSION_CUSTOM_LAST_IND = EMISSION_SPECID0;
const uint CONDUCTOR_COLOR = 0;
const uint CONDUCTOR_COLOR_LAST_IND = CONDUCTOR_COLOR;
const uint CONDUCTOR_ROUGH_U = UINT_MAIN_LAST_IND + 0;
const uint CONDUCTOR_ROUGH_V = UINT_MAIN_LAST_IND + 1;
const uint CONDUCTOR_ETA = UINT_MAIN_LAST_IND + 2;
const uint CONDUCTOR_K = UINT_MAIN_LAST_IND + 3;
const uint CONDUCTOR_TEXID0 = UINT_MAIN_LAST_IND + 4;
const uint CONDUCTOR_ETA_SPECID = UINT_MAIN_LAST_IND + 5;
const uint CONDUCTOR_K_SPECID = UINT_MAIN_LAST_IND + 6;
const uint CONDUCTOR_CUSTOM_LAST_IND = CONDUCTOR_K_SPECID;
const uint DIFFUSE_COLOR = 0;
const uint DIFFUSE_COLOR_LAST_IND = DIFFUSE_COLOR;
const uint DIFFUSE_ROUGHNESS = UINT_MAIN_LAST_IND + 0;
const uint DIFFUSE_TEXID0 = UINT_MAIN_LAST_IND + 1;
const uint DIFFUSE_SPECID = UINT_MAIN_LAST_IND + 2;
const uint DIFFUSE_CUSTOM_LAST_IND = DIFFUSE_SPECID;
const uint COLOR_DATA_SIZE = 3;
const uint CUSTOM_DATA_SIZE = 12;
struct Material
{
  vec4 colors[COLOR_DATA_SIZE]; ///< colors data

  vec4 row0[1];     ///< texture matrix
  vec4 row1[1];     ///< texture matrix
      
  float data[CUSTOM_DATA_SIZE]; ///< float, uint and custom data. Read uint: uint x = as_uint(data[INDEX]), write: data[INDEX] = as_float(x)
};
const uint BUILD_LOW = 0;
const uint BUILD_MEDIUM = 1;
const uint BUILD_HIGH = 2;
const uint BUILD_REFIT = 3;
struct CRT_Hit 
{
  float    t;         ///< intersection distance from ray origin to object
  uint primId; 
  uint instId;
  uint geomId;    ///< use 4 most significant bits for geometry type; thay are zero for triangles 
  float    coords[4]; ///< custom intersection data; for triangles coords[0] and coords[1] stores baricentric coords (u,v)
};
struct RefractResultT
{
  vec3 rayDir;
  bool   success;
  float  eta;

};
#define RefractResult RefractResultT
const uint INTEGRATOR_STUPID_PT = 0;
const uint INTEGRATOR_SHADOW_PT = 1;
const uint INTEGRATOR_MIS_PT = 2;

#ifndef SKIP_UBO_INCLUDE
#include "include/Integrator_generated_ubo.h"
#endif

/////////////////////////////////////////////////////////////////////
/////////////////// local functions /////////////////////////////////
/////////////////////////////////////////////////////////////////////

mat4 translate4x4(vec3 delta)
{
  return mat4(vec4(1.0, 0.0, 0.0, 0.0),
              vec4(0.0, 1.0, 0.0, 0.0),
              vec4(0.0, 0.0, 1.0, 0.0),
              vec4(delta, 1.0));
}

mat4 rotate4x4X(float phi)
{
  return mat4(vec4(1.0f, 0.0f,  0.0f,           0.0f),
              vec4(0.0f, +cos(phi),  +sin(phi), 0.0f),
              vec4(0.0f, -sin(phi),  +cos(phi), 0.0f),
              vec4(0.0f, 0.0f,       0.0f,      1.0f));
}

mat4 rotate4x4Y(float phi)
{
  return mat4(vec4(+cos(phi), 0.0f, -sin(phi), 0.0f),
              vec4(0.0f,      1.0f, 0.0f,      0.0f),
              vec4(+sin(phi), 0.0f, +cos(phi), 0.0f),
              vec4(0.0f,      0.0f, 0.0f,      1.0f));
}

mat4 rotate4x4Z(float phi)
{
  return mat4(vec4(+cos(phi), sin(phi), 0.0f, 0.0f),
              vec4(-sin(phi), cos(phi), 0.0f, 0.0f),
              vec4(0.0f,      0.0f,     1.0f, 0.0f),
              vec4(0.0f,      0.0f,     0.0f, 1.0f));
}

mat4 inverse4x4(mat4 m) { return inverse(m); }
vec3 mul4x3(mat4 m, vec3 v) { return (m*vec4(v, 1.0f)).xyz; }
vec3 mul3x3(mat4 m, vec3 v) { return (m*vec4(v, 0.0f)).xyz; }

float misHeuristicPower1(float p) { return isfinite(p) ? abs(p) : 0.0f; }

float epsilonOfPos(vec3 hitPos) { return max(max(abs(hitPos.x), max(abs(hitPos.y), abs(hitPos.z))), 2.0f*GEPSILON)*GEPSILON; }

uint NextState(inout RandomGen gen) {
  const uint x = (gen.state).x * 17 + (gen.state).y * 13123;
  (gen.state).x = (x << 13) ^ x;
  (gen.state).y ^= (x << 7);
  return x;
}

vec3 OffsRayPos(const vec3 a_hitPos, const vec3 a_surfaceNorm, const vec3 a_sampleDir) {
  const float signOfNormal2 = dot(a_sampleDir, a_surfaceNorm) < 0.0f ? -1.0f : 1.0f;
  const float offsetEps     = epsilonOfPos(a_hitPos);
  return a_hitPos + signOfNormal2*offsetEps*a_surfaceNorm;
}

float misWeightHeuristic(float a, float b) {
  const float w = misHeuristicPower1(a) / max(misHeuristicPower1(a) + misHeuristicPower1(b), 1e-30f);
  return isfinite(w) ? w : 0.0f;
}

float maxcomp(vec3 v) { return max(v.x, max(v.y, v.z)); }

float rndFloat1_Pseudo(inout RandomGen gen) {
  const uint x = NextState(gen);
  const uint tmp = (x * (x * x * 15731 + 74323) + 871483);
  const float scale      = (1.0f / 4294967296.0f);
  return (float((tmp)))*scale;
}

vec2 rndFloat2_Pseudo(inout RandomGen gen) {
  uint x = NextState(gen); 

  const uint x1 = (x * (x * x * 15731 + 74323) + 871483);
  const uint y1 = (x * (x * x * 13734 + 37828) + 234234);

  const float scale     = (1.0f / 4294967296.0f);

  return vec2(float((x1)), float((y1)))*scale;
}

float PdfWtoA(const float aPdfW, const float aDist, const float aCosThere) {
  return aPdfW * abs(aCosThere) / max(aDist*aDist, 1e-30f);
}

vec4 rndFloat4_Pseudo(inout RandomGen gen) {
  uint x = NextState(gen);

  const uint x1 = (x * (x * x * 15731 + 74323) + 871483);
  const uint y1 = (x * (x * x * 13734 + 37828) + 234234);
  const uint z1 = (x * (x * x * 11687 + 26461) + 137589);
  const uint w1 = (x * (x * x * 15707 + 789221) + 1376312589);

  const float scale = (1.0f / 4294967296.0f);

  return vec4(float((x1)), float((y1)), float((z1)), float((w1)))*scale;
}

uint fakeOffset(uint x, uint y, uint pitch) { return y*pitch + x; }  // RTV pattern, for 2D threading

#define KGEN_FLAG_RETURN            1
#define KGEN_FLAG_BREAK             2
#define KGEN_FLAG_DONT_SET_EXIT     4
#define KGEN_FLAG_SET_EXIT_NEGATIVE 8
#define KGEN_REDUCTION_LAST_STEP    16
#define RTC_RANDOM 
#define RTC_MATERIAL 
#define TEST_CLASS_H 
#define BASIC_PROJ_LOGIC_H 
#define MAXFLOAT FLT_MAX
#define CFLOAT_GUARDIAN 
#define IMAGE2D_H 
#define SPECTRUM_H 

