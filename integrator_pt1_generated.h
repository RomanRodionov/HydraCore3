#ifndef MAIN_CLASS_DECL_Integrator_H
#define MAIN_CLASS_DECL_Integrator_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

#include "vk_pipeline.h"
#include "vk_buffers.h"
#include "vk_utils.h"
#include "vk_copy.h"
#include "vk_context.h"


#include "integrator_pt.h"


#include "include/Integrator_generated_ubo.h"

class Integrator_Generated : public Integrator
{
public:

  Integrator_Generated(int a_maxThreads, _Bool a_spectral_mode) : Integrator(a_maxThreads, a_spectral_mode) 
  {
  }
  virtual void InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount);
  virtual void SetVulkanContext(vk_utils::VulkanContext a_ctx) { m_ctx = a_ctx; }

  virtual void SetVulkanInOutFor_RayTrace(
    VkBuffer out_colorBuffer,
    size_t   out_colorOffset,
    uint32_t dummyArgument = 0)
  {
    RayTrace_local.out_colorBuffer = out_colorBuffer;
    RayTrace_local.out_colorOffset = out_colorOffset;
    InitAllGeneratedDescriptorSets_RayTrace();
  }

  virtual void SetVulkanInOutFor_CastSingleRay(
    VkBuffer out_colorBuffer,
    size_t   out_colorOffset,
    uint32_t dummyArgument = 0)
  {
    CastSingleRay_local.out_colorBuffer = out_colorBuffer;
    CastSingleRay_local.out_colorOffset = out_colorOffset;
    InitAllGeneratedDescriptorSets_CastSingleRay();
  }

  virtual void SetVulkanInOutFor_PathTrace(
    VkBuffer out_colorBuffer,
    size_t   out_colorOffset,
    uint32_t dummyArgument = 0)
  {
    PathTrace_local.out_colorBuffer = out_colorBuffer;
    PathTrace_local.out_colorOffset = out_colorOffset;
    InitAllGeneratedDescriptorSets_PathTrace();
  }

  virtual void SetVulkanInOutFor_NaivePathTrace(
    VkBuffer out_colorBuffer,
    size_t   out_colorOffset,
    uint32_t dummyArgument = 0)
  {
    NaivePathTrace_local.out_colorBuffer = out_colorBuffer;
    NaivePathTrace_local.out_colorOffset = out_colorOffset;
    InitAllGeneratedDescriptorSets_NaivePathTrace();
  }

  virtual ~Integrator_Generated();


  virtual void InitMemberBuffers();
  virtual void UpdateAll(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine)
  {
    UpdatePlainMembers(a_pCopyEngine);
    UpdateVectorMembers(a_pCopyEngine);
    UpdateTextureMembers(a_pCopyEngine);
  }
  
  virtual void CommitDeviceData() override // you have to define this virtual function in the original imput class
  {
    InitMemberBuffers();
    UpdateAll(m_ctx.pCopyHelper);
  }  
  void GetExecutionTime(const char* a_funcName, float a_out[4]) override; 
  void UpdateMembersPlainData() override { UpdatePlainMembers(m_ctx.pCopyHelper); } 
  
  virtual void UpdatePlainMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  virtual void UpdateVectorMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  virtual void UpdateTextureMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  virtual void ReadPlainMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  
  virtual void RayTraceCmd(VkCommandBuffer a_commandBuffer, uint tid, float4* out_color);
  virtual void CastSingleRayCmd(VkCommandBuffer a_commandBuffer, uint tid, uint* out_color);
  virtual void PathTraceCmd(VkCommandBuffer a_commandBuffer, uint tid, float4* out_color);
  virtual void NaivePathTraceCmd(VkCommandBuffer a_commandBuffer, uint tid, float4* out_color);

  void RayTraceBlock(uint tid, float4* out_color, uint32_t a_numPasses) override;
  void CastSingleRayBlock(uint tid, uint* out_color, uint32_t a_numPasses) override;
  void PathTraceBlock(uint tid, float4* out_color, uint32_t a_numPasses) override;
  void NaivePathTraceBlock(uint tid, float4* out_color, uint32_t a_numPasses) override;

  inline vk_utils::ExecTime GetRayTraceExecutionTime() const { return m_exTimeRayTrace; }
  inline vk_utils::ExecTime GetCastSingleRayExecutionTime() const { return m_exTimeCastSingleRay; }
  inline vk_utils::ExecTime GetPathTraceExecutionTime() const { return m_exTimePathTrace; }
  inline vk_utils::ExecTime GetNaivePathTraceExecutionTime() const { return m_exTimeNaivePathTrace; }

  vk_utils::ExecTime m_exTimeRayTrace;
  vk_utils::ExecTime m_exTimeCastSingleRay;
  vk_utils::ExecTime m_exTimePathTrace;
  vk_utils::ExecTime m_exTimeNaivePathTrace;

  virtual void copyKernelFloatCmd(uint32_t length);
  
  virtual void RayTraceMegaCmd(uint tid, float4* out_color);
  virtual void CastSingleRayMegaCmd(uint tid, uint* out_color);
  virtual void PathTraceMegaCmd(uint tid, float4* out_color);
  virtual void NaivePathTraceMegaCmd(uint tid, float4* out_color);
  
  struct MemLoc
  {
    VkDeviceMemory memObject = VK_NULL_HANDLE;
    size_t         memOffset = 0;
    size_t         allocId   = 0;
  };

  virtual MemLoc AllocAndBind(const std::vector<VkBuffer>& a_buffers); ///< replace this function to apply custom allocator
  virtual MemLoc AllocAndBind(const std::vector<VkImage>& a_image);    ///< replace this function to apply custom allocator
  virtual void   FreeAllAllocations(std::vector<MemLoc>& a_memLoc);    ///< replace this function to apply custom allocator

protected:

  VkPhysicalDevice           physicalDevice = VK_NULL_HANDLE;
  VkDevice                   device         = VK_NULL_HANDLE;
  vk_utils::VulkanContext    m_ctx          = {};
  VkCommandBuffer            m_currCmdBuffer   = VK_NULL_HANDLE;
  uint32_t                   m_currThreadFlags = 0;
  std::vector<MemLoc>        m_allMems;
  VkPhysicalDeviceProperties m_devProps;

  VkBufferMemoryBarrier BarrierForClearFlags(VkBuffer a_buffer);
  VkBufferMemoryBarrier BarrierForSingleBuffer(VkBuffer a_buffer);
  void BarriersForSeveralBuffers(VkBuffer* a_inBuffers, VkBufferMemoryBarrier* a_outBarriers, uint32_t a_buffersNum);

  virtual void InitHelpers();
  virtual void InitBuffers(size_t a_maxThreadsCount, bool a_tempBuffersOverlay = true);
  virtual void InitKernels(const char* a_filePath);
  virtual void AllocateAllDescriptorSets();

  virtual void InitAllGeneratedDescriptorSets_RayTrace();
  virtual void InitAllGeneratedDescriptorSets_CastSingleRay();
  virtual void InitAllGeneratedDescriptorSets_PathTrace();
  virtual void InitAllGeneratedDescriptorSets_NaivePathTrace();

  virtual void AssignBuffersToMemory(const std::vector<VkBuffer>& a_buffers, VkDeviceMemory a_mem);

  virtual void AllocMemoryForMemberBuffersAndImages(const std::vector<VkBuffer>& a_buffers, const std::vector<VkImage>& a_image);
  virtual std::string AlterShaderPath(const char* in_shaderPath) { return std::string("") + std::string(in_shaderPath); }

  
  

  struct RayTrace_Data
  {
    VkBuffer out_colorBuffer = VK_NULL_HANDLE;
    size_t   out_colorOffset = 0;
    bool needToClearOutput = true;
  } RayTrace_local;

  struct CastSingleRay_Data
  {
    VkBuffer out_colorBuffer = VK_NULL_HANDLE;
    size_t   out_colorOffset = 0;
    bool needToClearOutput = true;
  } CastSingleRay_local;

  struct PathTrace_Data
  {
    VkBuffer out_colorBuffer = VK_NULL_HANDLE;
    size_t   out_colorOffset = 0;
    bool needToClearOutput = true;
  } PathTrace_local;

  struct NaivePathTrace_Data
  {
    VkBuffer out_colorBuffer = VK_NULL_HANDLE;
    size_t   out_colorOffset = 0;
    bool needToClearOutput = true;
  } NaivePathTrace_local;



  struct MembersDataGPU
  {
  } m_vdata;
  
  
  size_t m_maxThreadCount = 0;
  VkBuffer m_classDataBuffer = VK_NULL_HANDLE;

  VkPipelineLayout      RayTraceMegaLayout   = VK_NULL_HANDLE;
  VkPipeline            RayTraceMegaPipeline = VK_NULL_HANDLE; 
  VkDescriptorSetLayout RayTraceMegaDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateRayTraceMegaDSLayout();
  virtual void InitKernel_RayTraceMega(const char* a_filePath);
  VkPipelineLayout      CastSingleRayMegaLayout   = VK_NULL_HANDLE;
  VkPipeline            CastSingleRayMegaPipeline = VK_NULL_HANDLE; 
  VkDescriptorSetLayout CastSingleRayMegaDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateCastSingleRayMegaDSLayout();
  virtual void InitKernel_CastSingleRayMega(const char* a_filePath);
  VkPipelineLayout      PathTraceMegaLayout   = VK_NULL_HANDLE;
  VkPipeline            PathTraceMegaPipeline = VK_NULL_HANDLE; 
  VkDescriptorSetLayout PathTraceMegaDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreatePathTraceMegaDSLayout();
  virtual void InitKernel_PathTraceMega(const char* a_filePath);
  VkPipelineLayout      NaivePathTraceMegaLayout   = VK_NULL_HANDLE;
  VkPipeline            NaivePathTraceMegaPipeline = VK_NULL_HANDLE; 
  VkDescriptorSetLayout NaivePathTraceMegaDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateNaivePathTraceMegaDSLayout();
  virtual void InitKernel_NaivePathTraceMega(const char* a_filePath);


  virtual VkBufferUsageFlags GetAdditionalFlagsForUBO() const;
  virtual uint32_t           GetDefaultMaxTextures() const;

  VkPipelineLayout      copyKernelFloatLayout   = VK_NULL_HANDLE;
  VkPipeline            copyKernelFloatPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout copyKernelFloatDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreatecopyKernelFloatDSLayout();

  VkDescriptorPool m_dsPool = VK_NULL_HANDLE;
  VkDescriptorSet  m_allGeneratedDS[4];

  Integrator_Generated_UBO_Data m_uboData;
  
  constexpr static uint32_t MEMCPY_BLOCK_SIZE = 256;
  constexpr static uint32_t REDUCTION_BLOCK_SIZE = 256;

  virtual void MakeComputePipelineAndLayout(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout, 
                                            VkPipelineLayout* pPipelineLayout, VkPipeline* pPipeline);
  virtual void MakeComputePipelineOnly(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout, VkPipelineLayout pipelineLayout, 
                                       VkPipeline* pPipeline);

  std::vector<VkPipelineLayout> m_allCreatedPipelineLayouts; ///<! remenber them here to delete later
  std::vector<VkPipeline>       m_allCreatedPipelines;       ///<! remenber them here to delete later
};

#endif

