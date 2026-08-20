// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include <vk_descriptors.h>
#include <camera.h>
#include<vk_loader.h>



struct MeshAsset;

struct GPUSceneData {
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewproj;
	glm::mat4 lightViewProj; // area-light 2D shadow projection
	glm::vec4 ambientColor; // rgb = color, a = intensity
	glm::vec4 pointLightPosition; // xyz = point light, w = intensity
	glm::vec4 pointLightColor;
	glm::vec4 areaLightPosition; // xyz = area-light center, w = intensity
	glm::vec4 areaLightColor;
	glm::vec4 shadowParams; // x = far, y = point bias, z = area projective bias
	glm::vec4 cameraPosition; // xyz = world position
	glm::vec4 specularParams; // x = ks, y = shininess, z = roughness influence, w = IBL intensity
	glm::vec4 pcssParams; // x = light width, y = light height, z = max penumbra UV, w = near
};

struct ShadowUBO {
	glm::mat4 lightViewProj;
	glm::vec4 lightPosFar; // xyz = world position, w = far plane
};
struct EngineStats {
	float frametime;
	int triangle_count;
	int drawcall_count;
	float scene_update_time;
	float mesh_draw_time;
};

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		// reverse iterate the deletion queue to execute all the functions
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)(); //call functors
		}

		deletors.clear();
	}
};

struct FrameData{
	VkSemaphore _swapchainSemaphore, _renderSemaphore;
	VkFence _renderFence;
	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;
	DeletionQueue _deletionQueue;
	DescriptorAllocatorGrowable _frameDescriptors;
};
struct ComputePushConstants {
	glm::vec4 data1;
	glm::vec4 data2;
	glm::vec4 data3;
	glm::vec4 data4;
};
struct ComputeEffect {
	const char* name;

	VkPipeline pipeline;
	VkPipelineLayout layout;

	ComputePushConstants data;
};

class VulkanEngine;

struct GLTFMetallic_Roughness {
	MaterialPipeline opaquePipeline;
	MaterialPipeline transparentPipeline;

	VkDescriptorSetLayout materialLayout;

	struct MaterialConstants {
		glm::vec4 colorFactors;
		glm::vec4 metal_rough_factors;
		glm::vec4 emissiveFactor;
		glm::vec4 extra[13];
	};

	struct MaterialResources {
		AllocatedImage colorImage;
		VkSampler colorSampler;
		AllocatedImage metalRoughImage;
		VkSampler metalRoughSampler;
		AllocatedImage emissiveImage;
		VkSampler emissiveSampler;
		AllocatedImage normalImage;
		VkSampler normalSampler;
		VkBuffer dataBuffer;
		uint32_t dataBufferOffset;
	};

	DescriptorWriter writer;

	void build_pipelines(VulkanEngine* engine);
	void clear_resources(VkDevice device);

	MaterialInstance write_material(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator);
};

struct MeshNode : public Node {

	std::shared_ptr<MeshAsset> mesh;

	virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) override;
};

struct RenderObject {
	uint32_t indexCount;
	uint32_t firstIndex;
	VkBuffer indexBuffer;

	MaterialInstance* material;
	Bounds bounds;

	glm::mat4 transform;
	VkDeviceAddress vertexBufferAddress;
};

struct DrawContext {
	std::vector<RenderObject> OpaqueSurfaces;
	std::vector<RenderObject> TransparentSurfaces;
};



constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine {
public:

	bool _isInitialized{ false };
	int _frameNumber {0};
	bool stop_rendering{ false };
	VkExtent2D _windowExtent{ 1700 , 900 };

	struct SDL_Window* _window{ nullptr };

	VkInstance _instance;
	VkDebugUtilsMessengerEXT _debugMessenger;
	VkPhysicalDevice _chosenGPU;
	VkDevice _device;
	VkSurfaceKHR _surface;

	AllocatedImage _drawImage;
	AllocatedImage _tonemapImage;
	AllocatedImage _depthImage;

	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;

	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;
	VkExtent2D _swapchainExtent;

	FrameData _frames[FRAME_OVERLAP];

	FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; };

	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	DeletionQueue _mainDeletionQueue;

	VmaAllocator _allocator;

	DescriptorAllocatorGrowable globalDescriptorAllocator;

	VkDescriptorSet _drawImageDescriptors;
	VkDescriptorSetLayout _drawImageDescriptorLayout;

	VkPipeline _gradientPipeline;
	VkPipelineLayout _gradientPipelineLayout;

	VkFence _immFence;
	VkCommandBuffer _immCommandBuffer;
	VkCommandPool _immCommandPool;

	std::vector<ComputeEffect> backgroundEffects;
	int currentBackgroundEffect{ 0 };


	VkPipelineLayout _meshPipelineLayout;
	VkPipeline _meshPipeline;


	std::vector<std::shared_ptr<MeshAsset>> testMeshes;

	bool resize_requested;
	VkExtent2D _drawExtent;
	float renderScale = 1.f;

	GPUSceneData sceneData;
	glm::vec3 ambientLightColor{ 1.f, 1.f, 1.f };
	float ambientIntensity = 0.04f;
	glm::vec3 pointLightPos{ 2.f, 1.5f, 2.f };
	glm::vec3 pointLightCol{ 1.f, 1.f, 1.f };
	float pointLightIntensity = 8.f;
	float specularStrength = 0.45f;
	float specularShininess = 64.f;
	float specularRoughnessInfluence = 1.f;
	float areaLightWidth = 0.47f;
	float areaLightHeight = 0.39f;
	float areaLightIntensity = 0.f;
	float pcssMaxPenumbra = 0.06f;
	float iblIntensity = 1.f;
	float exposure = 1.f;

	VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

	AllocatedImage _shadowMap;
	VkSampler _shadowSampler;
	VkExtent2D _shadowMapExtent{ 1024, 1024 };
	AllocatedImage _shadowCubemap;
	std::array<VkImageView, 6> _shadowCubeFaceViews{};
	VkSampler _shadowCubeSampler;
	VkExtent2D _shadowCubeExtent{ 512, 512 };
	VkPipeline _shadowPipeline;
	VkPipeline _shadowCubePipeline;
	VkPipelineLayout _shadowPipelineLayout;
	VkDescriptorSetLayout _shadowDescriptorLayout;


	AllocatedImage _whiteImage;
	AllocatedImage _blackImage;
	AllocatedImage _greyImage;
	AllocatedImage _errorCheckerboardImage;
	AllocatedImage _flatNormalImage;

	VkSampler _defaultSamplerLinear;
	VkSampler _defaultSamplerNearest;

	VkDescriptorSetLayout _singleImageDescriptorLayout;


	MaterialInstance defaultData;
	GLTFMetallic_Roughness metalRoughMaterial; //test


	DrawContext mainDrawContext;
	std::unordered_map<std::string, std::shared_ptr<Node>> loadedNodes;

	Camera mainCamera;

	std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loadedScenes;

	EngineStats stats;

	void update_scene();





	void init_mesh_pipeline();


	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

	static VulkanEngine& Get();



	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();

	GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

	AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false, size_t bytesPerPixel = 4);
	AllocatedImage create_cubemap(uint32_t extent, VkFormat format, VkImageUsageFlags usage, uint32_t mipLevels = 1);
	void destroy_image(const AllocatedImage& img);
	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	void destroy_buffer(const AllocatedBuffer& buffer);

private:
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();
	void draw_background(VkCommandBuffer cmd);

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
	void init_descriptors();
	void init_pipelines();
	void init_background_pipelines();
	void init_tonemap_pipeline();
	void draw_tonemap(VkCommandBuffer cmd);
	void init_imgui();
	void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
	void draw_geometry(VkCommandBuffer cmd);
	void draw_shadows(VkCommandBuffer cmd);
	void draw_point_light_shadows(VkCommandBuffer cmd);
	void draw_area_light_shadows(VkCommandBuffer cmd);
	void init_shadow_pipeline();
	void init_shadow_map();
	void init_ibl();
	bool bake_irradiance();
	bool bake_prefiltered();
	bool bake_brdf_lut();
	void draw_skybox(VkCommandBuffer cmd);
	void init_default_data();
	void resize_swapchain();

	AllocatedImage _envCubemap{};
	AllocatedImage _irradianceCubemap{};
	AllocatedImage _prefilteredCubemap{};
	AllocatedImage _brdfLut{};
	VkSampler _iblCubeSampler{ VK_NULL_HANDLE };
	VkSampler _iblLutSampler{ VK_NULL_HANDLE };
	VkDescriptorSetLayout _skyboxDescriptorLayout{ VK_NULL_HANDLE };
	VkPipelineLayout _skyboxPipelineLayout{ VK_NULL_HANDLE };
	VkPipeline _skyboxPipeline{ VK_NULL_HANDLE };
	VkDescriptorSet _skyboxDescriptorSet{ VK_NULL_HANDLE };
	bool _iblReady{ false };

	VkDescriptorSetLayout _tonemapDescriptorLayout{ VK_NULL_HANDLE };
	VkPipelineLayout _tonemapPipelineLayout{ VK_NULL_HANDLE };
	VkPipeline _tonemapPipeline{ VK_NULL_HANDLE };
	VkDescriptorSet _tonemapDescriptorSet{ VK_NULL_HANDLE };
	VkSampler _tonemapSampler{ VK_NULL_HANDLE };
};
