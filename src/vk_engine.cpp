//> includes
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vk_initializers.h>
#include <vk_types.h>
#include <vk_images.h>
#include <VkBootstrap.h>

#include <chrono>
#include <thread>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#include <vk_pipelines.h>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include "stb_image.h"

#include <cmath>
#include <cstring>
#include <vector>



VulkanEngine* loadedEngine = nullptr;

constexpr bool bUseValidationLayers = false;

VulkanEngine& VulkanEngine::Get() { return *loadedEngine; }

bool is_visible(const RenderObject& obj, const glm::mat4& viewproj) {
    // Test the local AABB against frustum planes in clip space.
    // Perspective-dividing the 8 corners first fails for large meshes
    // (e.g. the Cornell ground plane): mixed/negative w flips NDC and
    // the object can be culled while still covering the screen.
    glm::mat4 matrix = viewproj * obj.transform;

    auto row = [&](int r) {
        return glm::vec4(matrix[0][r], matrix[1][r], matrix[2][r], matrix[3][r]);
    };

    const glm::vec4 planes[6] = {
        row(3) + row(0),
        row(3) - row(0),
        row(3) + row(1),
        row(3) - row(1),
        row(2),
        row(3) - row(2),
    };

    const glm::vec3 center = obj.bounds.origin;
    const glm::vec3 extent = obj.bounds.extents;

    for (const glm::vec4& plane : planes) {
        float dist = glm::dot(glm::vec3(plane), center) + plane.w;
        float radius = glm::dot(extent, glm::abs(glm::vec3(plane)));
        if (dist + radius < 0.f) {
            return false;
        }
    }
    return true;
}
void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    _window = SDL_CreateWindow(
        "Vulkan Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width,
        _windowExtent.height,
        window_flags);

	init_vulkan();
	init_swapchain();
	init_commands();
	init_sync_structures();
    init_descriptors();
    init_pipelines();
    init_imgui();

    init_default_data();
    mainCamera.velocity = glm::vec3(0.f);
    mainCamera.position = glm::vec3(0.f, 0.4f, 3.2f);
    mainCamera.pitch = 0;
    mainCamera.yaw = 0;

    std::string helmetPath = { "..\\assets\\DamagedHelmet.glb" };
    auto helmetFile = loadGltf(this, helmetPath);
    assert(helmetFile.has_value());
    loadedScenes["helmet"] = *helmetFile;

    // hide cursor and report unbounded relative motion for FPS look
    SDL_SetRelativeMouseMode(SDL_TRUE);

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::init_pipelines()
{
    init_background_pipelines();
    init_tonemap_pipeline();
    init_mesh_pipeline();
    init_shadow_pipeline();
    metalRoughMaterial.build_pipelines(this);
    _mainDeletionQueue.push_function([&]() {
        metalRoughMaterial.clear_resources(_device);
        });
}

void VulkanEngine::init_background_pipelines()
{
    VkPipelineLayoutCreateInfo computeLayout{};
    computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computeLayout.pNext = nullptr;
    computeLayout.pSetLayouts = &_drawImageDescriptorLayout;
    computeLayout.setLayoutCount = 1;

    VkPushConstantRange pushConstant{};
    pushConstant.offset = 0;
    pushConstant.size = sizeof(ComputePushConstants);
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    computeLayout.pPushConstantRanges = &pushConstant;
    computeLayout.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &computeLayout, nullptr, &_gradientPipelineLayout));

    //layout code
    VkShaderModule computeDrawShader;
    if (!vkutil::load_shader_module("../shaders/gradient_color.comp.spv", _device, &computeDrawShader))
    {
        fmt::print("Error when building the compute shader \n");
    }

    VkShaderModule skyShader;
    if (!vkutil::load_shader_module("../shaders/sky.comp.spv", _device, &skyShader)) {
        fmt::print("Error when building the compute shader \n");
    }


    VkPipelineShaderStageCreateInfo stageinfo{};
    stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageinfo.pNext = nullptr;
    stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageinfo.module = computeDrawShader;
    stageinfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.pNext = nullptr;
    computePipelineCreateInfo.layout = _gradientPipelineLayout;
    computePipelineCreateInfo.stage = stageinfo;

    ComputeEffect gradient;
    gradient.layout = _gradientPipelineLayout;
    gradient.name = "gradient";
    gradient.data = {};

    //default colors
    gradient.data.data1 = glm::vec4(1, 0, 0, 1);
    gradient.data.data2 = glm::vec4(0, 0, 1, 1);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradient.pipeline));

    //change the shader module only to create the sky shader
    computePipelineCreateInfo.stage.module = skyShader;

    ComputeEffect sky;
    sky.layout = _gradientPipelineLayout;
    sky.name = "sky";
    sky.data = {};
    //default sky parameters
    sky.data.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &sky.pipeline));
    //add the 2 background effects into the array
    backgroundEffects.push_back(gradient);
    backgroundEffects.push_back(sky);

    //destroy structures properly
    vkDestroyShaderModule(_device, computeDrawShader, nullptr);
    vkDestroyShaderModule(_device, skyShader, nullptr);
    _mainDeletionQueue.push_function([=]() {
        vkDestroyPipelineLayout(_device, _gradientPipelineLayout, nullptr);
        vkDestroyPipeline(_device, sky.pipeline, nullptr);
        vkDestroyPipeline(_device, gradient.pipeline, nullptr);
        });
}

void VulkanEngine::init_tonemap_pipeline()
{
    VkSamplerCreateInfo samplerInfo{ .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(_device, &samplerInfo, nullptr, &_tonemapSampler));

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    _tonemapDescriptorLayout = layoutBuilder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

    VkPushConstantRange push{};
    push.offset = 0;
    push.size = sizeof(float);
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipelineLayoutCreateInfo layoutInfo = vkinit::pipeline_layout_create_info();
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &_tonemapDescriptorLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &push;
    VK_CHECK(vkCreatePipelineLayout(_device, &layoutInfo, nullptr, &_tonemapPipelineLayout));

    VkShaderModule shader = VK_NULL_HANDLE;
    if (!vkutil::load_shader_module("../shaders/tonemap.comp.spv", _device, &shader)) {
        fmt::println("failed to load tonemap.comp.spv");
        return;
    }

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shader;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = _tonemapPipelineLayout;
    pipelineInfo.stage = stageInfo;
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &_tonemapPipeline));
    vkDestroyShaderModule(_device, shader, nullptr);

    _tonemapDescriptorSet = globalDescriptorAllocator.allocate(_device, _tonemapDescriptorLayout);
    DescriptorWriter writer;
    writer.write_image(0, _drawImage.imageView, _tonemapSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(1, _tonemapImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    writer.update_set(_device, _tonemapDescriptorSet);

    _mainDeletionQueue.push_function([this]() {
        vkDestroyPipeline(_device, _tonemapPipeline, nullptr);
        vkDestroyPipelineLayout(_device, _tonemapPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _tonemapDescriptorLayout, nullptr);
        vkDestroySampler(_device, _tonemapSampler, nullptr);
        });
}

void VulkanEngine::draw_tonemap(VkCommandBuffer cmd)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _tonemapPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _tonemapPipelineLayout, 0, 1, &_tonemapDescriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, _tonemapPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &exposure);
    vkCmdDispatch(cmd, (_drawExtent.width + 15) / 16, (_drawExtent.height + 15) / 16, 1);
}

void VulkanEngine::init_vulkan()
{
	vkb::InstanceBuilder builder;
    auto ret = builder.set_app_name("Vulkan Engine")
        .set_engine_name("Vulkan Engine")
        .require_api_version(1, 3, 0)
		.request_validation_layers(bUseValidationLayers)
        .use_default_debug_messenger()
		.build();

	vkb::Instance vkb_instance = ret.value();

	_instance = vkb_instance.instance;
	_debugMessenger = vkb_instance.debug_messenger;

	SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

	VkPhysicalDeviceVulkan13Features features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features.dynamicRendering = true;
	features.synchronization2 = true;

	VkPhysicalDeviceVulkan12Features features12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
	features12.bufferDeviceAddress = true;
	features12.descriptorIndexing = true;

	vkb::PhysicalDeviceSelector selector{ vkb_instance };
    vkb::PhysicalDevice physicalDevice = selector.set_surface(_surface)
		.set_minimum_version(1, 3)
		.set_required_features_13(features)
        .set_required_features_12(features12)
		.select()
		.value();
    
	vkb::DeviceBuilder deviceBuilder{ physicalDevice };

	vkb::Device vkb_device = deviceBuilder.build().value();

	_device = vkb_device.device;
	_chosenGPU = physicalDevice.physical_device;

    _graphicsQueue = vkb_device.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = _chosenGPU;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &_allocator);

    _mainDeletionQueue.push_function([&]() {
        vmaDestroyAllocator(_allocator);
        });
}


void VulkanEngine::init_swapchain()
{
    create_swapchain(_windowExtent.width, _windowExtent.height);
    //draw image size will match the window
    VkExtent3D drawImageExtent = {
        _windowExtent.width,
        _windowExtent.height,
        1
    };

    //hardcoding the draw format to 32 bit float
    _drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    _drawImage.imageExtent = drawImageExtent;

    VkImageUsageFlags drawImageUsages{};
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_SAMPLED_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo rimg_info = vkinit::image_create_info(_drawImage.imageFormat, drawImageUsages, drawImageExtent);

    //for the draw image, we want to allocate it from gpu local memory
    VmaAllocationCreateInfo rimg_allocinfo = {};
    rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    //allocate and create the image
    vmaCreateImage(_allocator, &rimg_info, &rimg_allocinfo, &_drawImage.image, &_drawImage.allocation, nullptr);

    //build a image-view for the draw image to use for rendering
    VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(_drawImage.imageFormat, _drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(_device, &rview_info, nullptr, &_drawImage.imageView));

    _depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
    _depthImage.imageExtent = drawImageExtent;
    VkImageUsageFlags depthImageUsages{};
    depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthImage.imageFormat, depthImageUsages, drawImageExtent);

    //allocate and create the image
    vmaCreateImage(_allocator, &dimg_info, &rimg_allocinfo, &_depthImage.image, &_depthImage.allocation, nullptr);

    //build a image-view for the draw image to use for rendering
    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthImage.imageFormat, _depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);

    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImage.imageView));

    _tonemapImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    _tonemapImage.imageExtent = drawImageExtent;
    VkImageUsageFlags tonemapUsages = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkImageCreateInfo timg_info = vkinit::image_create_info(_tonemapImage.imageFormat, tonemapUsages, drawImageExtent);
    VK_CHECK(vmaCreateImage(_allocator, &timg_info, &rimg_allocinfo, &_tonemapImage.image, &_tonemapImage.allocation, nullptr));
    VkImageViewCreateInfo tview_info = vkinit::imageview_create_info(_tonemapImage.imageFormat, _tonemapImage.image, VK_IMAGE_ASPECT_COLOR_BIT);
    VK_CHECK(vkCreateImageView(_device, &tview_info, nullptr, &_tonemapImage.imageView));

    //add to deletion queues
    _mainDeletionQueue.push_function([=]() {
        vkDestroyImageView(_device, _drawImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _drawImage.image, _drawImage.allocation);

        vkDestroyImageView(_device, _tonemapImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _tonemapImage.image, _tonemapImage.allocation);

        vkDestroyImageView(_device, _depthImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _depthImage.image, _depthImage.allocation);
        });
}

void VulkanEngine::init_commands()
{
    VkCommandPoolCreateInfo commandPoolInfo = {};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.pNext = nullptr;
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolInfo.queueFamilyIndex = _graphicsQueueFamily;

    for (int i = 0; i < FRAME_OVERLAP; i++) {

        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));

        // allocate the default command buffer that we will use for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = {};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.pNext = nullptr;
        cmdAllocInfo.commandPool = _frames[i]._commandPool;
        cmdAllocInfo.commandBufferCount = 1;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));
    }
    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_immCommandPool));

    // allocate the command buffer for immediate submits
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_immCommandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_immCommandBuffer));

    _mainDeletionQueue.push_function([=]() {
        vkDestroyCommandPool(_device, _immCommandPool, nullptr);
        });
    
}

void VulkanEngine::init_sync_structures()
{
    //create syncronization structures
    //one fence to control when the gpu has finished rendering the frame,
    //and 2 semaphores to syncronize rendering with swapchain
    //we want the fence to start signalled so we can wait on it on the first frame
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._swapchainSemaphore));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));
    }
    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
    _mainDeletionQueue.push_function([=]() { vkDestroyFence(_device, _immFence, nullptr); });
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
	vkb::SwapchainBuilder swapchainBuilder{ _chosenGPU, _device,  _surface };
	_swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    vkb::Swapchain vkb_swapchain = swapchainBuilder
        .set_desired_format(VkSurfaceFormatKHR{ .format = _swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
		.add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
		.value();

    _swapchainExtent = vkb_swapchain.extent;
    //store swapchain and its related images
    _swapchain = vkb_swapchain.swapchain;
    _swapchainImages = vkb_swapchain.get_images().value();
    _swapchainImageViews = vkb_swapchain.get_image_views().value();
}

void VulkanEngine::destroy_swapchain()
{
	vkDestroySwapchainKHR(_device, _swapchain, nullptr);

    for(auto imageView : _swapchainImageViews) {
        vkDestroyImageView(_device, imageView, nullptr);
	}
}


void VulkanEngine::cleanup()
{

    if (_isInitialized) {

        vkDeviceWaitIdle(_device);
        loadedScenes.clear();
        for(int i = 0; i < FRAME_OVERLAP; i++) {
            vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);

            //destroy sync objects
            vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
            vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
            vkDestroySemaphore(_device, _frames[i]._swapchainSemaphore, nullptr);

            _frames[i]._deletionQueue.flush();
		}
        for (auto& mesh : testMeshes) {
            destroy_buffer(mesh->meshBuffers.indexBuffer);
            destroy_buffer(mesh->meshBuffers.vertexBuffer);
        }
		_mainDeletionQueue.flush();

        destroy_swapchain();

        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkDestroyDevice(_device, nullptr);

        vkb::destroy_debug_utils_messenger(_instance, _debugMessenger);
        vkDestroyInstance(_instance, nullptr);
        SDL_DestroyWindow(_window);

    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::draw_background(VkCommandBuffer cmd)
{


    ComputeEffect& effect = backgroundEffects[currentBackgroundEffect];

    // bind the background compute pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);

    // bind the descriptor set containing the draw image for the compute pipeline
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _gradientPipelineLayout, 0, 1, &_drawImageDescriptors, 0, nullptr);

    vkCmdPushConstants(cmd, _gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &effect.data);
    // execute the compute pipeline dispatch. We are using 16x16 workgroup size so we need to divide by it
    vkCmdDispatch(cmd, std::ceil(_drawExtent.width / 16.0), std::ceil(_drawExtent.height / 16.0), 1);

}

void VulkanEngine::draw()
{
    update_scene();

    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, 1000000000));

    get_current_frame()._deletionQueue.flush();
    get_current_frame()._frameDescriptors.clear_pools(_device);

    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

    uint32_t swapchainImageIndex;
    VkResult e = vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame()._swapchainSemaphore, nullptr, &swapchainImageIndex);
    if (e == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;
        return;
    }

    //naming it cmd for shorter writing
    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

    // now that we are sure that the commands finished executing, we can safely
    // reset the command buffer to begin recording again.
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    //begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    _drawExtent.height = std::min(_swapchainExtent.height, _drawImage.imageExtent.height) * renderScale;
    _drawExtent.width = std::min(_swapchainExtent.width, _drawImage.imageExtent.width) * renderScale;

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // transition our main draw image into general layout so we can write into it
    // we will overwrite it all so we dont care about what was the older layout
    if (_iblReady) {
        vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
    else {
        vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        draw_background(cmd);
        vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
    vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    vkutil::transition_image(cmd, _shadowCubemap.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    vkutil::transition_image(cmd, _shadowMap.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    draw_shadows(cmd);
    vkutil::transition_image(cmd, _shadowCubemap.image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
    vkutil::transition_image(cmd, _shadowMap.image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);

    draw_geometry(cmd);

    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    vkutil::transition_image(cmd, _tonemapImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    draw_tonemap(cmd);
    vkutil::transition_image(cmd, _tonemapImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkutil::copy_image_to_image(cmd, _tonemapImage.image, _swapchainImages[swapchainImageIndex], _drawExtent, _swapchainExtent);

    // set swapchain image layout to Present so we can show it on the screen
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

	draw_imgui(cmd, _swapchainImageViews[swapchainImageIndex]);

    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    //finalize the command buffer (we can no longer add commands, but it can now be executed)
    VK_CHECK(vkEndCommandBuffer(cmd));

    //prepare the submission to the queue. 
    //we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
    //we will signal the _renderSemaphore, to signal that rendering has finished

    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);

    VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, get_current_frame()._swapchainSemaphore);
    VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, get_current_frame()._renderSemaphore);

    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, &signalInfo, &waitInfo);

    //submit command buffer to the queue and execute it.
    // _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, get_current_frame()._renderFence));

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.pSwapchains = &_swapchain;
    presentInfo.swapchainCount = 1;

    presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    VkResult presentResult = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;
    }

    //increase the number of frames drawn
    _frameNumber++;

}

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while (!bQuit) {
        //begin clock
        auto start = std::chrono::system_clock::now();
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT)
                bQuit = true;

            // Esc toggles mouse capture so ImGui / OS cursor remain usable
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) {
                SDL_SetRelativeMouseMode(SDL_GetRelativeMouseMode() ? SDL_FALSE : SDL_TRUE);
            }

            mainCamera.processSDLEvent(e);
            ImGui_ImplSDL2_ProcessEvent(&e);

            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    stop_rendering = false;
                }
            }
        }

        // do not draw if we are minimized
        if (stop_rendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (resize_requested) {
            resize_swapchain();
        }

        // imgui new frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Stats");

        ImGui::Text("frametime %f ms", stats.frametime);
        ImGui::Text("draw time %f ms", stats.mesh_draw_time);
        ImGui::Text("update time %f ms", stats.scene_update_time);
        ImGui::Text("triangles %i", stats.triangle_count);
        ImGui::Text("draws %i", stats.drawcall_count);
        ImGui::End();

        if (ImGui::Begin("Lighting")) {
            ImGui::ColorEdit3("Ambient Color", &ambientLightColor.x);
            ImGui::SliderFloat("Ambient Intensity", &ambientIntensity, 0.f, 2.f);
            ImGui::SliderFloat("IBL Intensity", &iblIntensity, 0.f, 4.f);
            ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.f);
            ImGui::Separator();
            ImGui::Text("Point Light");
            ImGui::DragFloat3("Point Position", &pointLightPos.x, 0.05f);
            ImGui::ColorEdit3("Point Color", &pointLightCol.x);
            ImGui::SliderFloat("Point Intensity", &pointLightIntensity, 0.f, 40.f);
            ImGui::Separator();
            ImGui::Text("Area Light PCSS");
            ImGui::SliderFloat("Area Intensity", &areaLightIntensity, 0.f, 80.f);
            ImGui::SliderFloat("Light Width", &areaLightWidth, 0.f, 2.f);
            ImGui::SliderFloat("Light Height", &areaLightHeight, 0.f, 2.f);
            ImGui::SliderFloat("Max Penumbra", &pcssMaxPenumbra, 0.001f, 0.2f);
            ImGui::Separator();
            ImGui::SliderFloat("Specular Strength", &specularStrength, 0.f, 2.f);
            ImGui::SliderFloat("Shininess", &specularShininess, 1.f, 256.f);
            ImGui::SliderFloat("Roughness Influence", &specularRoughnessInfluence, 0.f, 1.f);
        }
        ImGui::End();

        ImGui::Render();


        draw();

        //get clock again, compare with start clock
        auto end = std::chrono::system_clock::now();

        //convert to microseconds (integer), and then come back to miliseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        stats.frametime = elapsed.count() / 1000.f;
    }
}

void VulkanEngine::init_descriptors()
{
    // global allocator: long-lived sets (draw image + materials). Must include
    // all descriptor types those layouts use; Growable can add pools later.
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes =
    {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8 },
    };

    globalDescriptorAllocator.init(_device, 10, sizes);

    //make the descriptor set layout for our compute draw

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE); 
        _drawImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT); 
    }
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        builder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        builder.add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        builder.add_binding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        builder.add_binding(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        _gpuSceneDataDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        _singleImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    //allocate a descriptor set for our draw image
    _drawImageDescriptors = globalDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);

    DescriptorWriter writer;
    writer.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    writer.update_set(_device, _drawImageDescriptors);

    //make sure both the descriptor allocator and the new layout get cleaned up properly
    _mainDeletionQueue.push_function([&]() {
        globalDescriptorAllocator.destroy_pools(_device);

        vkDestroyDescriptorSetLayout(_device, _drawImageDescriptorLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _gpuSceneDataDescriptorLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _singleImageDescriptorLayout, nullptr);
        });

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        // create a descriptor pool
        std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frame_sizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8 },
        };

        _frames[i]._frameDescriptors = DescriptorAllocatorGrowable{};
        _frames[i]._frameDescriptors.init(_device, 1000, frame_sizes);

        _mainDeletionQueue.push_function([&, i]() {
            _frames[i]._frameDescriptors.destroy_pools(_device);
            });
    }
}

void VulkanEngine::init_mesh_pipeline()
{
    VkShaderModule triangleFragShader;
    if (!vkutil::load_shader_module("../shaders/tex_image.frag.spv", _device, &triangleFragShader)) {
        fmt::print("Error when building the fragment shader \n");
    }
    else {
        fmt::print("Triangle fragment shader succesfully loaded \n");
    }

    VkShaderModule triangleVertexShader;
    if (!vkutil::load_shader_module("../shaders/colored_triangle_mesh.vert.spv", _device, &triangleVertexShader)) {
        fmt::print("Error when building the vertex shader \n");
    }
    else {
        fmt::print("Triangle vertex shader succesfully loaded \n");
    }

    VkPushConstantRange bufferRange{};
    bufferRange.offset = 0;
    bufferRange.size = sizeof(GPUDrawPushConstants);
    bufferRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    pipeline_layout_info.pPushConstantRanges = &bufferRange;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pSetLayouts = &_singleImageDescriptorLayout;
    pipeline_layout_info.setLayoutCount = 1;
    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_meshPipelineLayout));


    PipelineBuilder pipelineBuilder;

    //use the triangle layout we created
    pipelineBuilder._pipelineLayout = _meshPipelineLayout;
    //connecting the vertex and pixel shaders to the pipeline
    pipelineBuilder.set_shaders(triangleVertexShader, triangleFragShader);
    //it will draw triangles
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    //filled triangles
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    //no backface culling
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    //no multisampling
    pipelineBuilder.set_multisampling_none();
    //no blending
    //pipelineBuilder.disable_blending();
    pipelineBuilder.disable_blending();

    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

    //connect the image format we will draw into, from draw image
    pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(_depthImage.imageFormat);

    //finally build the pipeline
    _meshPipeline = pipelineBuilder.build_pipeline(_device);

    //clean structures
    vkDestroyShaderModule(_device, triangleFragShader, nullptr);
    vkDestroyShaderModule(_device, triangleVertexShader, nullptr);

    _mainDeletionQueue.push_function([&]() {
        vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
        vkDestroyPipeline(_device, _meshPipeline, nullptr);
        });

}


void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    VK_CHECK(vkResetFences(_device, 1, &_immFence));
    VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

    VkCommandBuffer cmd = _immCommandBuffer;

    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, nullptr, nullptr);

    // submit command buffer to the queue and execute it.
    //  _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));

    VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, 9999999999));
}

void VulkanEngine::init_imgui()
{
    // 1: create descriptor pool for IMGUI
    //  the size of the pool is very oversize, but it's copied from imgui demo
    //  itself.
    VkDescriptorPoolSize pool_sizes[] = { { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 } };

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000;
    pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;

    VkDescriptorPool imguiPool;
    VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));

    // 2: initialize imgui library

    // this initializes the core structures of imgui
    ImGui::CreateContext();

    // this initializes imgui for SDL
    ImGui_ImplSDL2_InitForVulkan(_window);

    // this initializes imgui for Vulkan
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = _instance;
    init_info.PhysicalDevice = _chosenGPU;
    init_info.Device = _device;
    init_info.Queue = _graphicsQueue;
    init_info.DescriptorPool = imguiPool;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.UseDynamicRendering = true;

    //dynamic rendering parameters for imgui to use
    init_info.PipelineRenderingCreateInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &_swapchainImageFormat;


    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&init_info);

    ImGui_ImplVulkan_CreateFontsTexture();

    // add the destroy the imgui created structures
    _mainDeletionQueue.push_function([=]() {
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(_device, imguiPool, nullptr);
        });
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView)
{
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderInfo = vkinit::rendering_info(_swapchainExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

void VulkanEngine::draw_geometry(VkCommandBuffer cmd)
{
    //reset counters
    stats.drawcall_count = 0;
    stats.triangle_count = 0;
    //begin clock
    auto start = std::chrono::system_clock::now();
    std::vector<uint32_t> opaque_draws;
    opaque_draws.reserve(mainDrawContext.OpaqueSurfaces.size());

    for (int i = 0; i < mainDrawContext.OpaqueSurfaces.size(); i++) {
        if (is_visible(mainDrawContext.OpaqueSurfaces[i], sceneData.viewproj)) {
            opaque_draws.push_back(i);
        }
    }

    // sort the opaque surfaces by material and mesh
    std::sort(opaque_draws.begin(), opaque_draws.end(), [&](const auto& iA, const auto& iB) {
        const RenderObject& A = mainDrawContext.OpaqueSurfaces[iA];
        const RenderObject& B = mainDrawContext.OpaqueSurfaces[iB];
        if (A.material == B.material) {
            return A.indexBuffer < B.indexBuffer;
        }
        else {
            return A.material < B.material;
        }
        });

    AllocatedBuffer gpuSceneDataBuffer = create_buffer(sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    get_current_frame()._deletionQueue.push_function([=, this](){
        destroy_buffer(gpuSceneDataBuffer);
        });
    GPUSceneData* sceneUniformData =static_cast<GPUSceneData*>(gpuSceneDataBuffer.allocation->GetMappedData());

    *sceneUniformData = sceneData;
    VkDescriptorSet globalDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _gpuSceneDataDescriptorLayout);
    DescriptorWriter writer;

    writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.write_image(1, _shadowMap.imageView, _shadowSampler, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(2, _shadowCubemap.imageView, _shadowCubeSampler, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(3, _irradianceCubemap.imageView, _iblCubeSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(4, _prefilteredCubemap.image ? _prefilteredCubemap.imageView : _irradianceCubemap.imageView, _iblCubeSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(5, _brdfLut.imageView, _iblLutSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    writer.update_set(_device, globalDescriptor);

    //begin a render pass  connected to our draw image
    VkClearValue skyClear{ .color = { {0.f, 0.f, 0.f, 1.f} } };
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(
        _drawImage.imageView, _iblReady ? &skyClear : nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(_depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(_drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);

    if (_iblReady) {
        draw_skybox(cmd);
    }

    MaterialPipeline* lastPipeline = nullptr;
    MaterialInstance* lastMaterial = nullptr;
    VkBuffer lastIndexBuffer = VK_NULL_HANDLE;

    auto draw = [&](const RenderObject& r) {
        if (r.material != lastMaterial) {
            lastMaterial = r.material;
            //rebind pipeline and descriptors if the material changed
            if (r.material->pipeline != lastPipeline) {

                lastPipeline = r.material->pipeline;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 0, 1,
                    &globalDescriptor, 0, nullptr);

                VkViewport viewport = {};
                viewport.x = 0;
                viewport.y = 0;
                viewport.width = (float)_windowExtent.width;
                viewport.height = (float)_windowExtent.height;
                viewport.minDepth = 0.f;
                viewport.maxDepth = 1.f;

                vkCmdSetViewport(cmd, 0, 1, &viewport);

                VkRect2D scissor = {};
                scissor.offset.x = 0;
                scissor.offset.y = 0;
                scissor.extent.width = _windowExtent.width;
                scissor.extent.height = _windowExtent.height;

                vkCmdSetScissor(cmd, 0, 1, &scissor);
            }

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 1, 1,
                &r.material->materialSet, 0, nullptr);
        }
        //rebind index buffer if needed
        if (r.indexBuffer != lastIndexBuffer) {
            lastIndexBuffer = r.indexBuffer;
            vkCmdBindIndexBuffer(cmd, r.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }
        // calculate final mesh matrix
        GPUDrawPushConstants push_constants;
        push_constants.worldMatrix = r.transform;
        push_constants.vertexBuffer = r.vertexBufferAddress;

        vkCmdPushConstants(cmd, r.material->pipeline->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &push_constants);

        vkCmdDrawIndexed(cmd, r.indexCount, 1, r.firstIndex, 0, 0);
        //stats
        stats.drawcall_count++;
        stats.triangle_count += r.indexCount / 3;
        };

    for (auto& r : opaque_draws) {
        draw(mainDrawContext.OpaqueSurfaces[r]);
    }

    for (auto& r : mainDrawContext.TransparentSurfaces) {
        draw(r);
    }

    vkCmdEndRendering(cmd);

    auto end = std::chrono::system_clock::now();

    //convert to microseconds (integer), and then come back to miliseconds
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.mesh_draw_time = elapsed.count() / 1000.f;
}

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    // allocate buffer
    VkBufferCreateInfo bufferInfo = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.pNext = nullptr;
    bufferInfo.size = allocSize;

    bufferInfo.usage = usage;

    VmaAllocationCreateInfo vmaallocInfo = {};
    vmaallocInfo.usage = memoryUsage;
    vmaallocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    AllocatedBuffer newBuffer;

    // allocate the buffer
    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocInfo, &newBuffer.buffer, &newBuffer.allocation,
        &newBuffer.info));

    return newBuffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer& buffer)
{
    vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
}

GPUMeshBuffers VulkanEngine::uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices)
{
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;

    //create vertex buffer
    newSurface.vertexBuffer = create_buffer(vertexBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    //find the adress of the vertex buffer
    VkBufferDeviceAddressInfo deviceAdressInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,.buffer = newSurface.vertexBuffer.buffer };
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(_device, &deviceAdressInfo);

    //create index buffer
    newSurface.indexBuffer = create_buffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    AllocatedBuffer staging = create_buffer(vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    void* data = staging.allocation->GetMappedData();

    // copy vertex buffer
    memcpy(data, vertices.data(), vertexBufferSize);
    // copy index buffer
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    immediate_submit([&](VkCommandBuffer cmd) {
        VkBufferCopy vertexCopy{ 0 };
        vertexCopy.dstOffset = 0;
        vertexCopy.srcOffset = 0;
        vertexCopy.size = vertexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

        VkBufferCopy indexCopy{ 0 };
        indexCopy.dstOffset = 0;
        indexCopy.srcOffset = vertexBufferSize;
        indexCopy.size = indexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
        });

    destroy_buffer(staging);

    return newSurface;

}

void VulkanEngine::init_default_data() {

    testMeshes = loadGltfMeshes(this, "..\\assets\\basicmesh.glb").value();

    //3 default textures, white, grey, black. 1 pixel each
    uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
    _whiteImage = create_image((void*)&white, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1));
    _greyImage = create_image((void*)&grey, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
    _blackImage = create_image((void*)&black, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT);

    // tangent-space flat normal (0.5, 0.5, 1) so missing maps stay geometric
    uint32_t flatNormal = glm::packUnorm4x8(glm::vec4(0.5f, 0.5f, 1.f, 1.f));
    _flatNormalImage = create_image((void*)&flatNormal, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT);

    //checkerboard image
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16 > pixels; //for 16x16 checkerboard texture
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }
    _errorCheckerboardImage = create_image(pixels.data(), VkExtent3D{ 16, 16, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT);

    VkSamplerCreateInfo sampl = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };

    sampl.magFilter = VK_FILTER_NEAREST;
    sampl.minFilter = VK_FILTER_NEAREST;

    vkCreateSampler(_device, &sampl, nullptr, &_defaultSamplerNearest);

    sampl.magFilter = VK_FILTER_LINEAR;
    sampl.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(_device, &sampl, nullptr, &_defaultSamplerLinear);

    _mainDeletionQueue.push_function([&]() {
        vkDestroySampler(_device, _defaultSamplerNearest, nullptr);
        vkDestroySampler(_device, _defaultSamplerLinear, nullptr);

        destroy_image(_whiteImage);
        destroy_image(_greyImage);
        destroy_image(_blackImage);
        destroy_image(_flatNormalImage);
        destroy_image(_errorCheckerboardImage);
        });

    init_shadow_map();
    init_ibl();



    GLTFMetallic_Roughness::MaterialResources materialResources;
    //default the material textures
    materialResources.colorImage = _whiteImage;
    materialResources.colorSampler = _defaultSamplerLinear;
    materialResources.metalRoughImage = _whiteImage;
    materialResources.metalRoughSampler = _defaultSamplerLinear;
    materialResources.emissiveImage = _blackImage;
    materialResources.emissiveSampler = _defaultSamplerLinear;
    materialResources.normalImage = _flatNormalImage;
    materialResources.normalSampler = _defaultSamplerLinear;

    //set the uniform buffer for the material data
    AllocatedBuffer materialConstants = create_buffer(sizeof(GLTFMetallic_Roughness::MaterialConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    //write the buffer
    GLTFMetallic_Roughness::MaterialConstants* sceneUniformData = (GLTFMetallic_Roughness::MaterialConstants*)materialConstants.allocation->GetMappedData();
    sceneUniformData->colorFactors = glm::vec4{ 1,1,1,1 };
    sceneUniformData->metal_rough_factors = glm::vec4{ 1,0.5,0,0 };
    sceneUniformData->emissiveFactor = glm::vec4{ 0,0,0,0 };
    sceneUniformData->extra[0] = glm::vec4{ 1.f, 0.f, 0.f, 0.f };

    _mainDeletionQueue.push_function([=, this]() {
        destroy_buffer(materialConstants);
        });

    materialResources.dataBuffer = materialConstants.buffer;
    materialResources.dataBufferOffset = 0;

    defaultData = metalRoughMaterial.write_material(_device, MaterialPass::MainColor, materialResources, globalDescriptorAllocator);

    for (auto& m : testMeshes) {
        std::shared_ptr<MeshNode> newNode = std::make_shared<MeshNode>();
        newNode->mesh = m;

        newNode->localTransform = glm::mat4{ 1.f };
        newNode->worldTransform = glm::mat4{ 1.f };

        for (auto& s : newNode->mesh->surfaces) {
            s.material = std::make_shared<GLTFMaterial>(defaultData);
        }

        loadedNodes[m->name] = std::move(newNode);
    }
}

void VulkanEngine::resize_swapchain()
{
    vkDeviceWaitIdle(_device);

    destroy_swapchain();

    int w, h;
    SDL_GetWindowSize(_window, &w, &h);
    _windowExtent.width = w;
    _windowExtent.height = h;

    create_swapchain(_windowExtent.width, _windowExtent.height);

    resize_requested = false;
}

AllocatedImage VulkanEngine::create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = size;

    VkImageCreateInfo img_info = vkinit::image_create_info(format, usage, size);
    if (mipmapped) {
        img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
    }
    newImage.mipLevels = img_info.mipLevels;

    // always allocate images on dedicated GPU memory
    VmaAllocationCreateInfo allocinfo = {};
    allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // allocate and create the image
    VK_CHECK(vmaCreateImage(_allocator, &img_info, &allocinfo, &newImage.image, &newImage.allocation, nullptr));

    // if the format is a depth format, we will need to have it use the correct
    // aspect flag
    VkImageAspectFlags aspectFlag = VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_D32_SFLOAT) {
        aspectFlag = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    // build a image-view for the image
    VkImageViewCreateInfo view_info = vkinit::imageview_create_info(format, newImage.image, aspectFlag);
    view_info.subresourceRange.levelCount = img_info.mipLevels;

    VK_CHECK(vkCreateImageView(_device, &view_info, nullptr, &newImage.imageView));

    return newImage;
}

AllocatedImage VulkanEngine::create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped, size_t bytesPerPixel)
{
    if (bytesPerPixel == 0) {
        bytesPerPixel = 4;
    }
    size_t data_size = size.depth * size.width * size.height * bytesPerPixel;
    AllocatedBuffer uploadbuffer = create_buffer(data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    memcpy(uploadbuffer.info.pMappedData, data, data_size);

    AllocatedImage new_image = create_image(size, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mipmapped);

    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy copyRegion = {};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;

        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = size;

        // copy the buffer into the image
        vkCmdCopyBufferToImage(cmd, uploadbuffer.buffer, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
            &copyRegion);

        if (mipmapped) {
            vkutil::generate_mipmaps(cmd, new_image.image, VkExtent2D{ new_image.imageExtent.width,new_image.imageExtent.height });
        }
        else {
            vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        });
    destroy_buffer(uploadbuffer);
    return new_image;
}
void VulkanEngine::destroy_image(const AllocatedImage& img)
{
    vkDestroyImageView(_device, img.imageView, nullptr);
    vmaDestroyImage(_allocator, img.image, img.allocation);
}

AllocatedImage VulkanEngine::create_cubemap(uint32_t extent, VkFormat format, VkImageUsageFlags usage, uint32_t mipLevels)
{
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = { extent, extent, 1 };
    newImage.mipLevels = std::max(mipLevels, 1u);

    VkImageCreateInfo img_info = vkinit::image_create_info(format, usage, newImage.imageExtent);
    img_info.arrayLayers = 6;
    img_info.mipLevels = newImage.mipLevels;
    img_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

    VmaAllocationCreateInfo allocinfo = {};
    allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocinfo.preferredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // Depth shadow cubes keep dedicated allocation; color IBL cubes do not.
    if (format == VK_FORMAT_D32_SFLOAT) {
        allocinfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }

    VkResult cubemapAlloc = vmaCreateImage(_allocator, &img_info, &allocinfo, &newImage.image, &newImage.allocation, nullptr);
    if (cubemapAlloc != VK_SUCCESS) {
        fmt::println("create_cubemap failed: {} (extent {}, format {})", string_VkResult(cubemapAlloc), extent, (int)format);
        allocinfo.flags = 0;
        cubemapAlloc = vmaCreateImage(_allocator, &img_info, &allocinfo, &newImage.image, &newImage.allocation, nullptr);
    }
    VK_CHECK(cubemapAlloc);

    VkImageAspectFlags aspectFlag = (format == VK_FORMAT_D32_SFLOAT) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

    VkImageViewCreateInfo view_info = vkinit::imageview_create_info(format, newImage.image, aspectFlag);
    view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    view_info.subresourceRange.layerCount = 6;
    view_info.subresourceRange.levelCount = newImage.mipLevels;

    VK_CHECK(vkCreateImageView(_device, &view_info, nullptr, &newImage.imageView));
    return newImage;
}

static uint16_t float_to_half(float value)
{
    uint32_t f;
    std::memcpy(&f, &value, sizeof(f));
    const uint32_t sign = (f >> 16) & 0x8000;
    const int32_t exponent = int32_t((f >> 23) & 0xFF) - 127 + 15;
    const uint32_t mantissa = f & 0x7FFFFF;
    if (exponent <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    return static_cast<uint16_t>(sign | (uint32_t(exponent) << 10) | (mantissa >> 13));
}

bool VulkanEngine::bake_irradiance()
{
    const uint32_t extent = 32;
    AllocatedImage irradiance = create_cubemap(extent, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, 1);

    VkImageViewCreateInfo arrayViewInfo = vkinit::imageview_create_info(VK_FORMAT_R16G16B16A16_SFLOAT, irradiance.image, VK_IMAGE_ASPECT_COLOR_BIT);
    arrayViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    arrayViewInfo.subresourceRange.layerCount = 6;
    arrayViewInfo.subresourceRange.levelCount = 1;
    VkImageView arrayView = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(_device, &arrayViewInfo, nullptr, &arrayView));

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    VkDescriptorSetLayout setLayout = layoutBuilder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

    VkPipelineLayoutCreateInfo layoutInfo = vkinit::pipeline_layout_create_info();
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(_device, &layoutInfo, nullptr, &pipelineLayout));

    VkShaderModule shader = VK_NULL_HANDLE;
    if (!vkutil::load_shader_module("../shaders/ibl_irradiance.comp.spv", _device, &shader)) {
        fmt::println("IBL: failed to load ibl_irradiance.comp.spv");
        vkDestroyPipelineLayout(_device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, setLayout, nullptr);
        vkDestroyImageView(_device, arrayView, nullptr);
        destroy_image(irradiance);
        return false;
    }

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shader;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.stage = stageInfo;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));
    vkDestroyShaderModule(_device, shader, nullptr);

    VkDescriptorSet set = globalDescriptorAllocator.allocate(_device, setLayout);
    DescriptorWriter writer;
    writer.write_image(0, _envCubemap.imageView, _iblCubeSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(1, arrayView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    writer.update_set(_device, set);

    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, irradiance.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
        vkCmdDispatch(cmd, (extent + 7) / 8, (extent + 7) / 8, 6);

        VkImageMemoryBarrier2 barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.image = irradiance.image;
        barrier.subresourceRange = vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
        VkDependencyInfo dep{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
        });

    vkDestroyPipeline(_device, pipeline, nullptr);
    vkDestroyPipelineLayout(_device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(_device, setLayout, nullptr);
    vkDestroyImageView(_device, arrayView, nullptr);

    destroy_image(_irradianceCubemap);
    _irradianceCubemap = irradiance;
    fmt::println("IBL: irradiance cubemap ready ({}x{})", extent, extent);
    return true;
}

bool VulkanEngine::bake_prefiltered()
{
    const uint32_t extent = 128;
    const uint32_t mipCount = uint32_t(std::floor(std::log2(float(extent)))) + 1;
    AllocatedImage prefiltered = create_cubemap(extent, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, mipCount);

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    VkDescriptorSetLayout setLayout = layoutBuilder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

    VkPushConstantRange push{};
    push.offset = 0;
    push.size = sizeof(float) * 2 + sizeof(uint32_t) * 2;
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipelineLayoutCreateInfo layoutInfo = vkinit::pipeline_layout_create_info();
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &push;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(_device, &layoutInfo, nullptr, &pipelineLayout));

    VkShaderModule shader = VK_NULL_HANDLE;
    if (!vkutil::load_shader_module("../shaders/ibl_prefilter.comp.spv", _device, &shader)) {
        fmt::println("IBL: failed to load ibl_prefilter.comp.spv");
        vkDestroyPipelineLayout(_device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, setLayout, nullptr);
        destroy_image(prefiltered);
        return false;
    }

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shader;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.stage = stageInfo;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));
    vkDestroyShaderModule(_device, shader, nullptr);

    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, prefiltered.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        });

    struct PrefilterPC {
        float roughness;
        float srcResolution;
        uint32_t sampleCount;
        uint32_t pad;
    };

    std::vector<VkImageView> mipViews(mipCount, VK_NULL_HANDLE);
    for (uint32_t mip = 0; mip < mipCount; ++mip) {
        VkImageViewCreateInfo viewInfo = vkinit::imageview_create_info(VK_FORMAT_R16G16B16A16_SFLOAT, prefiltered.image, VK_IMAGE_ASPECT_COLOR_BIT);
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        viewInfo.subresourceRange.baseMipLevel = mip;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 6;
        VK_CHECK(vkCreateImageView(_device, &viewInfo, nullptr, &mipViews[mip]));

        VkDescriptorSet set = globalDescriptorAllocator.allocate(_device, setLayout);
        DescriptorWriter writer;
        writer.write_image(0, _envCubemap.imageView, _iblCubeSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writer.write_image(1, mipViews[mip], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        writer.update_set(_device, set);

        const uint32_t mipExtent = std::max(extent >> mip, 1u);
        PrefilterPC pc{};
        pc.roughness = (mipCount > 1) ? float(mip) / float(mipCount - 1) : 0.f;
        pc.srcResolution = 512.f;
        pc.sampleCount = (mip < 3) ? 64u : 128u;

        immediate_submit([&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
            vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PrefilterPC), &pc);
            vkCmdDispatch(cmd, (mipExtent + 7) / 8, (mipExtent + 7) / 8, 6);
            });
    }

    immediate_submit([&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier2 barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.image = prefiltered.image;
        barrier.subresourceRange = vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
        VkDependencyInfo dep{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
        });

    for (VkImageView view : mipViews) {
        vkDestroyImageView(_device, view, nullptr);
    }
    vkDestroyPipeline(_device, pipeline, nullptr);
    vkDestroyPipelineLayout(_device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(_device, setLayout, nullptr);

    if (_prefilteredCubemap.image) {
        destroy_image(_prefilteredCubemap);
    }
    _prefilteredCubemap = prefiltered;
    fmt::println("IBL: prefiltered cubemap ready ({}x{}, {} mips)", extent, extent, mipCount);
    return true;
}

bool VulkanEngine::bake_brdf_lut()
{
    const uint32_t extent = 256;
    AllocatedImage lut = create_image(VkExtent3D{ extent, extent, 1 }, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    VkDescriptorSetLayout setLayout = layoutBuilder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

    VkPipelineLayoutCreateInfo layoutInfo = vkinit::pipeline_layout_create_info();
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(_device, &layoutInfo, nullptr, &pipelineLayout));

    VkShaderModule shader = VK_NULL_HANDLE;
    if (!vkutil::load_shader_module("../shaders/ibl_brdf_lut.comp.spv", _device, &shader)) {
        fmt::println("IBL: failed to load ibl_brdf_lut.comp.spv");
        vkDestroyPipelineLayout(_device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, setLayout, nullptr);
        destroy_image(lut);
        return false;
    }

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shader;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.stage = stageInfo;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));
    vkDestroyShaderModule(_device, shader, nullptr);

    VkDescriptorSet set = globalDescriptorAllocator.allocate(_device, setLayout);
    DescriptorWriter writer;
    writer.write_image(0, lut.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    writer.update_set(_device, set);

    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, lut.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
        vkCmdDispatch(cmd, (extent + 7) / 8, (extent + 7) / 8, 1);

        VkImageMemoryBarrier2 barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.image = lut.image;
        barrier.subresourceRange = vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
        VkDependencyInfo dep{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
        });

    vkDestroyPipeline(_device, pipeline, nullptr);
    vkDestroyPipelineLayout(_device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(_device, setLayout, nullptr);

    destroy_image(_brdfLut);
    _brdfLut = lut;
    fmt::println("IBL: BRDF LUT ready ({}x{})", extent, extent);
    return true;
}

void VulkanEngine::init_ibl()
{
    _irradianceCubemap = create_cubemap(1, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT, 1);
    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, _irradianceCubemap.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });

    VkSamplerCreateInfo cubeSamplerInfo{ .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    cubeSamplerInfo.magFilter = VK_FILTER_LINEAR;
    cubeSamplerInfo.minFilter = VK_FILTER_LINEAR;
    cubeSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    cubeSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    cubeSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    cubeSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    cubeSamplerInfo.minLod = 0.f;
    cubeSamplerInfo.maxLod = 16.f;
    VK_CHECK(vkCreateSampler(_device, &cubeSamplerInfo, nullptr, &_iblCubeSampler));

    VkSamplerCreateInfo lutSamplerInfo{ .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    lutSamplerInfo.magFilter = VK_FILTER_LINEAR;
    lutSamplerInfo.minFilter = VK_FILTER_LINEAR;
    lutSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    lutSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    lutSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    lutSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(_device, &lutSamplerInfo, nullptr, &_iblLutSampler));

    uint16_t lutPixel[4] = { float_to_half(1.f), float_to_half(0.f), 0, float_to_half(1.f) };
    _brdfLut = create_image(lutPixel, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT, false, 8);

    _mainDeletionQueue.push_function([this]() {
        vkDestroySampler(_device, _iblCubeSampler, nullptr);
        vkDestroySampler(_device, _iblLutSampler, nullptr);
        destroy_image(_irradianceCubemap);
        destroy_image(_brdfLut);
        if (_prefilteredCubemap.image) {
            destroy_image(_prefilteredCubemap);
        }
        });

    int width = 0;
    int height = 0;
    int components = 0;
    float* hdr = stbi_loadf("..\\assets\\brown_photostudio_02_4k.hdr", &width, &height, &components, 3);
    if (!hdr) {
        fmt::println("IBL: failed to load HDR ({})", stbi_failure_reason());
        return;
    }
    fmt::println("IBL: loaded HDR {}x{} ({} ch)", width, height, components);

    std::vector<uint16_t> rgba16(size_t(width) * size_t(height) * 4);
    for (int i = 0; i < width * height; ++i) {
        rgba16[i * 4 + 0] = float_to_half(hdr[i * 3 + 0]);
        rgba16[i * 4 + 1] = float_to_half(hdr[i * 3 + 1]);
        rgba16[i * 4 + 2] = float_to_half(hdr[i * 3 + 2]);
        rgba16[i * 4 + 3] = float_to_half(1.f);
    }
    stbi_image_free(hdr);

    AllocatedImage equirect = create_image(rgba16.data(),
        VkExtent3D{ uint32_t(width), uint32_t(height), 1 },
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT,
        false,
        8);

    const uint32_t cubeExtent = 512;
    const uint32_t cubeMips = uint32_t(std::floor(std::log2(float(cubeExtent)))) + 1;
    _envCubemap = create_cubemap(cubeExtent, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        cubeMips);

    VkImageViewCreateInfo arrayViewInfo = vkinit::imageview_create_info(VK_FORMAT_R16G16B16A16_SFLOAT, _envCubemap.image, VK_IMAGE_ASPECT_COLOR_BIT);
    arrayViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    arrayViewInfo.subresourceRange.layerCount = 6;
    arrayViewInfo.subresourceRange.levelCount = 1;
    VkImageView cubeArrayView = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(_device, &arrayViewInfo, nullptr, &cubeArrayView));

    VkSamplerCreateInfo equirectSamplerInfo{ .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    equirectSamplerInfo.magFilter = VK_FILTER_LINEAR;
    equirectSamplerInfo.minFilter = VK_FILTER_LINEAR;
    equirectSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    equirectSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    equirectSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    equirectSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSampler equirectSampler = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSampler(_device, &equirectSamplerInfo, nullptr, &equirectSampler));

    DescriptorLayoutBuilder computeLayoutBuilder;
    computeLayoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    computeLayoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    VkDescriptorSetLayout computeSetLayout = computeLayoutBuilder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

    VkPipelineLayoutCreateInfo computeLayoutInfo = vkinit::pipeline_layout_create_info();
    computeLayoutInfo.setLayoutCount = 1;
    computeLayoutInfo.pSetLayouts = &computeSetLayout;
    VkPipelineLayout computeLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(_device, &computeLayoutInfo, nullptr, &computeLayout));

    VkShaderModule equirectShader = VK_NULL_HANDLE;
    if (!vkutil::load_shader_module("../shaders/ibl_equirect_to_cube.comp.spv", _device, &equirectShader)) {
        fmt::println("IBL: failed to load ibl_equirect_to_cube.comp.spv");
        vkDestroyPipelineLayout(_device, computeLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, computeSetLayout, nullptr);
        vkDestroySampler(_device, equirectSampler, nullptr);
        vkDestroyImageView(_device, cubeArrayView, nullptr);
        destroy_image(equirect);
        destroy_image(_envCubemap);
        return;
    }

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = equirectShader;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineInfo{};
    computePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineInfo.layout = computeLayout;
    computePipelineInfo.stage = stageInfo;
    VkPipeline computePipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineInfo, nullptr, &computePipeline));
    vkDestroyShaderModule(_device, equirectShader, nullptr);

    VkDescriptorSet computeSet = globalDescriptorAllocator.allocate(_device, computeSetLayout);
    DescriptorWriter computeWriter;
    computeWriter.write_image(0, equirect.imageView, equirectSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    computeWriter.write_image(1, cubeArrayView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    computeWriter.update_set(_device, computeSet);

    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, _envCubemap.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeLayout, 0, 1, &computeSet, 0, nullptr);
        vkCmdDispatch(cmd, (cubeExtent + 15) / 16, (cubeExtent + 15) / 16, 6);

        VkImageMemoryBarrier2 barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.image = _envCubemap.image;
        barrier.subresourceRange = vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
        VkDependencyInfo dep{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);

        vkutil::generate_cubemap_mipmaps(cmd, _envCubemap.image, VkExtent2D{ cubeExtent, cubeExtent }, cubeMips);
        });

    vkDestroyPipeline(_device, computePipeline, nullptr);
    vkDestroyPipelineLayout(_device, computeLayout, nullptr);
    vkDestroyDescriptorSetLayout(_device, computeSetLayout, nullptr);
    vkDestroySampler(_device, equirectSampler, nullptr);
    vkDestroyImageView(_device, cubeArrayView, nullptr);
    destroy_image(equirect);

    bake_irradiance();
    bake_prefiltered();
    bake_brdf_lut();

    DescriptorLayoutBuilder skyLayoutBuilder;
    skyLayoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    _skyboxDescriptorLayout = skyLayoutBuilder.build(_device, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPushConstantRange skyPush{};
    skyPush.offset = 0;
    skyPush.size = sizeof(glm::mat4) * 2;
    skyPush.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkPipelineLayoutCreateInfo skyLayoutInfo = vkinit::pipeline_layout_create_info();
    skyLayoutInfo.setLayoutCount = 1;
    skyLayoutInfo.pSetLayouts = &_skyboxDescriptorLayout;
    skyLayoutInfo.pushConstantRangeCount = 1;
    skyLayoutInfo.pPushConstantRanges = &skyPush;
    VK_CHECK(vkCreatePipelineLayout(_device, &skyLayoutInfo, nullptr, &_skyboxPipelineLayout));

    VkShaderModule skyVert = VK_NULL_HANDLE;
    VkShaderModule skyFrag = VK_NULL_HANDLE;
    if (!vkutil::load_shader_module("../shaders/skybox.vert.spv", _device, &skyVert)
        || !vkutil::load_shader_module("../shaders/skybox.frag.spv", _device, &skyFrag)) {
        fmt::println("IBL: failed to load skybox shaders");
        vkDestroyPipelineLayout(_device, _skyboxPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _skyboxDescriptorLayout, nullptr);
        destroy_image(_envCubemap);
        return;
    }

    PipelineBuilder skyBuilder;
    skyBuilder.set_shaders(skyVert, skyFrag);
    skyBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    skyBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    skyBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    skyBuilder.set_multisampling_none();
    skyBuilder.disable_blending();
    skyBuilder.disable_depthtest();
    skyBuilder.set_color_attachment_format(_drawImage.imageFormat);
    skyBuilder.set_depth_format(_depthImage.imageFormat);
    skyBuilder._pipelineLayout = _skyboxPipelineLayout;
    _skyboxPipeline = skyBuilder.build_pipeline(_device);
    vkDestroyShaderModule(_device, skyVert, nullptr);
    vkDestroyShaderModule(_device, skyFrag, nullptr);

    _skyboxDescriptorSet = globalDescriptorAllocator.allocate(_device, _skyboxDescriptorLayout);
    DescriptorWriter skyWriter;
    skyWriter.write_image(0, _envCubemap.imageView, _iblCubeSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    skyWriter.update_set(_device, _skyboxDescriptorSet);

    _mainDeletionQueue.push_function([this]() {
        vkDestroyPipeline(_device, _skyboxPipeline, nullptr);
        vkDestroyPipelineLayout(_device, _skyboxPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _skyboxDescriptorLayout, nullptr);
        destroy_image(_envCubemap);
        });

    _iblReady = true;
    fmt::println("IBL: environment cubemap ready ({} mips)", cubeMips);
}

void VulkanEngine::draw_skybox(VkCommandBuffer cmd)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _skyboxPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _skyboxPipelineLayout, 0, 1, &_skyboxDescriptorSet, 0, nullptr);

    VkViewport viewport{};
    viewport.width = (float)_drawExtent.width;
    viewport.height = (float)_drawExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = _drawExtent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    struct SkyboxPC {
        glm::mat4 view;
        glm::mat4 proj;
    } pc;
    pc.view = sceneData.view;
    pc.proj = sceneData.proj;
    vkCmdPushConstants(cmd, _skyboxPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(SkyboxPC), &pc);
    vkCmdDraw(cmd, 36, 1, 0, 0);
}

void VulkanEngine::init_shadow_map()
{
    _shadowMap = create_image(VkExtent3D{ _shadowMapExtent.width, _shadowMapExtent.height, 1 },
        VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    VkSamplerCreateInfo sampl{ .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sampl.magFilter = VK_FILTER_NEAREST;
    sampl.minFilter = VK_FILTER_NEAREST;
    sampl.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampl.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampl.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampl.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampl.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK(vkCreateSampler(_device, &sampl, nullptr, &_shadowSampler));

    _shadowCubemap = create_cubemap(_shadowCubeExtent.width, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    for (uint32_t i = 0; i < 6; i++) {
        VkImageViewCreateInfo viewInfo = vkinit::imageview_create_info(VK_FORMAT_D32_SFLOAT, _shadowCubemap.image, VK_IMAGE_ASPECT_DEPTH_BIT);
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.subresourceRange.baseArrayLayer = i;
        viewInfo.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(_device, &viewInfo, nullptr, &_shadowCubeFaceViews[i]));
    }

    VkSamplerCreateInfo cubeSampl{ .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    cubeSampl.magFilter = VK_FILTER_NEAREST;
    cubeSampl.minFilter = VK_FILTER_NEAREST;
    cubeSampl.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    cubeSampl.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    cubeSampl.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    cubeSampl.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(_device, &cubeSampl, nullptr, &_shadowCubeSampler));

    _mainDeletionQueue.push_function([&]() {
        vkDestroySampler(_device, _shadowSampler, nullptr);
        destroy_image(_shadowMap);
        vkDestroySampler(_device, _shadowCubeSampler, nullptr);
        for (VkImageView view : _shadowCubeFaceViews) {
            vkDestroyImageView(_device, view, nullptr);
        }
        destroy_image(_shadowCubemap);
        });
}

void VulkanEngine::init_shadow_pipeline()
{
    VkShaderModule cubeFragShader;
    if (!vkutil::load_shader_module("../shaders/shadow.frag.spv", _device, &cubeFragShader)) {
        fmt::println("Error when building the shadow fragment shader module");
    }
    VkShaderModule areaFragShader;
    if (!vkutil::load_shader_module("../shaders/shadow_2d.frag.spv", _device, &areaFragShader)) {
        fmt::println("Error when building the 2D shadow fragment shader module");
    }
    VkShaderModule shadowVertexShader;
    if (!vkutil::load_shader_module("../shaders/shadow.vert.spv", _device, &shadowVertexShader)) {
        fmt::println("Error when building the shadow vertex shader module");
    }

    VkPushConstantRange matrixRange{};
    matrixRange.offset = 0;
    matrixRange.size = sizeof(GPUDrawPushConstants);
    matrixRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    _shadowDescriptorLayout = layoutBuilder.build(_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineLayoutCreateInfo layoutInfo = vkinit::pipeline_layout_create_info();
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &_shadowDescriptorLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &matrixRange;
    VK_CHECK(vkCreatePipelineLayout(_device, &layoutInfo, nullptr, &_shadowPipelineLayout));

    PipelineBuilder pipelineBuilder;
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.disable_blending();
    pipelineBuilder.disable_color_attachment();
    pipelineBuilder.set_depth_format(VK_FORMAT_D32_SFLOAT);
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_LESS);
    pipelineBuilder._pipelineLayout = _shadowPipelineLayout;

    pipelineBuilder.set_shaders(shadowVertexShader, areaFragShader);
    _shadowPipeline = pipelineBuilder.build_pipeline(_device);

    pipelineBuilder.set_shaders(shadowVertexShader, cubeFragShader);
    _shadowCubePipeline = pipelineBuilder.build_pipeline(_device);

    vkDestroyShaderModule(_device, cubeFragShader, nullptr);
    vkDestroyShaderModule(_device, areaFragShader, nullptr);
    vkDestroyShaderModule(_device, shadowVertexShader, nullptr);

    _mainDeletionQueue.push_function([&]() {
        vkDestroyPipeline(_device, _shadowPipeline, nullptr);
        vkDestroyPipeline(_device, _shadowCubePipeline, nullptr);
        vkDestroyPipelineLayout(_device, _shadowPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _shadowDescriptorLayout, nullptr);
        });
}

void VulkanEngine::draw_shadows(VkCommandBuffer cmd)
{
    draw_point_light_shadows(cmd);
    draw_area_light_shadows(cmd);
}

void VulkanEngine::draw_area_light_shadows(VkCommandBuffer cmd)
{
    const glm::vec3 lightPos = glm::vec3(sceneData.areaLightPosition);
    const float farPlane = sceneData.shadowParams.x;

    AllocatedBuffer ubo = create_buffer(sizeof(ShadowUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    get_current_frame()._deletionQueue.push_function([=, this]() {
        destroy_buffer(ubo);
        });

    ShadowUBO* data = static_cast<ShadowUBO*>(ubo.allocation->GetMappedData());
    data->lightViewProj = sceneData.lightViewProj;
    data->lightPosFar = glm::vec4(lightPos, farPlane);

    VkDescriptorSet shadowSet = get_current_frame()._frameDescriptors.allocate(_device, _shadowDescriptorLayout);
    DescriptorWriter writer;
    writer.write_buffer(0, ubo.buffer, sizeof(ShadowUBO), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(_device, shadowSet);

    VkViewport viewport{};
    viewport.width = static_cast<float>(_shadowMapExtent.width);
    viewport.height = static_cast<float>(_shadowMapExtent.height);
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;

    VkRect2D scissor{};
    scissor.extent = _shadowMapExtent;

    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(
        _shadowMap.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, 1.f);
    VkRenderingInfo renderInfo = vkinit::rendering_info(_shadowMapExtent, nullptr, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _shadowPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _shadowPipelineLayout, 0, 1, &shadowSet, 0, nullptr);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    VkBuffer lastIndexBuffer = VK_NULL_HANDLE;
    for (const RenderObject& r : mainDrawContext.OpaqueSurfaces) {
        if (r.indexBuffer != lastIndexBuffer) {
            lastIndexBuffer = r.indexBuffer;
            vkCmdBindIndexBuffer(cmd, r.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }

        GPUDrawPushConstants push_constants;
        push_constants.worldMatrix = r.transform;
        push_constants.vertexBuffer = r.vertexBufferAddress;
        vkCmdPushConstants(cmd, _shadowPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &push_constants);
        vkCmdDrawIndexed(cmd, r.indexCount, 1, r.firstIndex, 0, 0);
    }

    vkCmdEndRendering(cmd);
}

void VulkanEngine::draw_point_light_shadows(VkCommandBuffer cmd)
{
    const glm::vec3 lightPos = glm::vec3(sceneData.pointLightPosition);
    const float farPlane = sceneData.shadowParams.x;
    const float nearPlane = 0.05f;
    glm::mat4 shadowProj = glm::perspective(glm::radians(90.0f), 1.0f, nearPlane, farPlane);

    const glm::vec3 targets[6] = {
        glm::vec3(1.f, 0.f, 0.f),
        glm::vec3(-1.f, 0.f, 0.f),
        glm::vec3(0.f, 1.f, 0.f),
        glm::vec3(0.f, -1.f, 0.f),
        glm::vec3(0.f, 0.f, 1.f),
        glm::vec3(0.f, 0.f, -1.f),
    };
    const glm::vec3 ups[6] = {
        glm::vec3(0.f, -1.f, 0.f),
        glm::vec3(0.f, -1.f, 0.f),
        glm::vec3(0.f, 0.f, 1.f),
        glm::vec3(0.f, 0.f, -1.f),
        glm::vec3(0.f, -1.f, 0.f),
        glm::vec3(0.f, -1.f, 0.f),
    };

    VkViewport viewport{};
    viewport.width = static_cast<float>(_shadowCubeExtent.width);
    viewport.height = static_cast<float>(_shadowCubeExtent.height);
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;

    VkRect2D scissor{};
    scissor.extent = _shadowCubeExtent;

    for (int face = 0; face < 6; face++) {
        AllocatedBuffer ubo = create_buffer(sizeof(ShadowUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        get_current_frame()._deletionQueue.push_function([=, this]() {
            destroy_buffer(ubo);
            });

        ShadowUBO* data = static_cast<ShadowUBO*>(ubo.allocation->GetMappedData());
        data->lightViewProj = shadowProj * glm::lookAt(lightPos, lightPos + targets[face], ups[face]);
        data->lightPosFar = glm::vec4(lightPos, farPlane);

        VkDescriptorSet shadowSet = get_current_frame()._frameDescriptors.allocate(_device, _shadowDescriptorLayout);
        DescriptorWriter writer;
        writer.write_buffer(0, ubo.buffer, sizeof(ShadowUBO), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.update_set(_device, shadowSet);

        VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(
            _shadowCubeFaceViews[face], VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, 1.f);
        VkRenderingInfo renderInfo = vkinit::rendering_info(_shadowCubeExtent, nullptr, &depthAttachment);
        vkCmdBeginRendering(cmd, &renderInfo);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _shadowCubePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _shadowPipelineLayout, 0, 1, &shadowSet, 0, nullptr);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        VkBuffer lastIndexBuffer = VK_NULL_HANDLE;
        for (const RenderObject& r : mainDrawContext.OpaqueSurfaces) {
            if (r.indexBuffer != lastIndexBuffer) {
                lastIndexBuffer = r.indexBuffer;
                vkCmdBindIndexBuffer(cmd, r.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            }

            GPUDrawPushConstants push_constants;
            push_constants.worldMatrix = r.transform;
            push_constants.vertexBuffer = r.vertexBufferAddress;
            vkCmdPushConstants(cmd, _shadowPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &push_constants);
            vkCmdDrawIndexed(cmd, r.indexCount, 1, r.firstIndex, 0, 0);
        }

        vkCmdEndRendering(cmd);
    }
}

void GLTFMetallic_Roughness::build_pipelines(VulkanEngine* engine)
{
    VkShaderModule meshFragShader;
    if (!vkutil::load_shader_module("../shaders/mesh.frag.spv", engine->_device, &meshFragShader)) {
        fmt::println("Error when building the triangle fragment shader module");
    }

    VkShaderModule meshVertexShader;
    if (!vkutil::load_shader_module("../shaders/mesh.vert.spv", engine->_device, &meshVertexShader)) {
        fmt::println("Error when building the triangle vertex shader module");
    }

    VkPushConstantRange matrixRange{};
    matrixRange.offset = 0;
    matrixRange.size = sizeof(GPUDrawPushConstants);
    matrixRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.add_binding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    materialLayout = layoutBuilder.build(engine->_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    VkDescriptorSetLayout layouts[] = { engine->_gpuSceneDataDescriptorLayout,
        materialLayout };

    VkPipelineLayoutCreateInfo mesh_layout_info = vkinit::pipeline_layout_create_info();
    mesh_layout_info.setLayoutCount = 2;
    mesh_layout_info.pSetLayouts = layouts;
    mesh_layout_info.pPushConstantRanges = &matrixRange;
    mesh_layout_info.pushConstantRangeCount = 1;

    VkPipelineLayout newLayout;
    VK_CHECK(vkCreatePipelineLayout(engine->_device, &mesh_layout_info, nullptr, &newLayout));

    opaquePipeline.layout = newLayout;
    transparentPipeline.layout = newLayout;

    // build the stage-create-info for both vertex and fragment stages. This lets
    // the pipeline know the shader modules per stage
    PipelineBuilder pipelineBuilder;
    pipelineBuilder.set_shaders(meshVertexShader, meshFragShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.disable_blending();
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

    //render format
    pipelineBuilder.set_color_attachment_format(engine->_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(engine->_depthImage.imageFormat);

    // use the triangle layout we created
    pipelineBuilder._pipelineLayout = newLayout;

    // finally build the pipeline
    opaquePipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    // create the transparent variant
    pipelineBuilder.enable_blending_additive();

    pipelineBuilder.enable_depthtest(false, VK_COMPARE_OP_GREATER_OR_EQUAL);

    transparentPipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    vkDestroyShaderModule(engine->_device, meshFragShader, nullptr);
    vkDestroyShaderModule(engine->_device, meshVertexShader, nullptr);
}

MaterialInstance GLTFMetallic_Roughness::write_material(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator)
{
    MaterialInstance matData;
    matData.passType = pass;
    if (pass == MaterialPass::Transparent) {
        matData.pipeline = &transparentPipeline;
    }
    else {
        matData.pipeline = &opaquePipeline;
    }

    matData.materialSet = descriptorAllocator.allocate(device, materialLayout);


    writer.clear();
    writer.write_buffer(0, resources.dataBuffer, sizeof(MaterialConstants), resources.dataBufferOffset, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.write_image(1, resources.colorImage.imageView, resources.colorSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(2, resources.metalRoughImage.imageView, resources.metalRoughSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(3, resources.emissiveImage.imageView, resources.emissiveSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(4, resources.normalImage.imageView, resources.normalSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    writer.update_set(device, matData.materialSet);

    return matData;
}

void GLTFMetallic_Roughness::clear_resources(VkDevice device)
{
    vkDestroyPipeline(device, opaquePipeline.pipeline, nullptr);
    vkDestroyPipeline(device, transparentPipeline.pipeline, nullptr);
    vkDestroyPipelineLayout(device, opaquePipeline.layout, nullptr);
    vkDestroyDescriptorSetLayout(device, materialLayout, nullptr);
}

void MeshNode::Draw(const glm::mat4& topMatrix, DrawContext& ctx)
{
    glm::mat4 nodeMatrix = topMatrix * worldTransform;

    for (auto& s : mesh->surfaces) {
        RenderObject def;
        def.indexCount = s.count;
        def.firstIndex = s.startIndex;
        def.indexBuffer = mesh->meshBuffers.indexBuffer.buffer;
        def.material = &s.material->data;
        def.bounds = s.bounds;
        def.transform = nodeMatrix;
        def.vertexBufferAddress = mesh->meshBuffers.vertexBufferAddress;

        if (s.material->data.passType == MaterialPass::Transparent) {
            ctx.TransparentSurfaces.push_back(def);
        }
        else {
            ctx.OpaqueSurfaces.push_back(def);
        }
    }

    // recurse down
    Node::Draw(topMatrix, ctx);
}

void VulkanEngine::update_scene()
{
    //begin clock
    auto start = std::chrono::system_clock::now();

    mainDrawContext.OpaqueSurfaces.clear();
    mainDrawContext.TransparentSurfaces.clear();

    // loadedNodes["Suzanne"]->Draw(glm::mat4{ 1.f }, mainDrawContext);
    loadedScenes["helmet"]->Draw(glm::mat4{ 1.f }, mainDrawContext);

    mainCamera.update();

    glm::mat4 view = mainCamera.getViewMatrix();

    // camera projection
    glm::mat4 projection = glm::perspective(glm::radians(70.f), (float)_windowExtent.width / (float)_windowExtent.height, 10000.f, 0.1f);

    // invert the Y direction on projection matrix so that we are more similar
    // to opengl and gltf axis
    projection[1][1] *= -1;

    sceneData.view = view;
    sceneData.proj = projection;
    sceneData.viewproj = projection * view;
    sceneData.cameraPosition = glm::vec4(mainCamera.position, 1.f);
    sceneData.ambientColor = glm::vec4(ambientLightColor, ambientIntensity);
    sceneData.specularParams = glm::vec4(specularStrength, specularShininess, specularRoughnessInfluence, iblIntensity);

    sceneData.pointLightPosition = glm::vec4(pointLightPos, pointLightIntensity);
    sceneData.pointLightColor = glm::vec4(pointLightCol, 1.f);

    glm::vec3 areaPos = glm::vec3(0.f, 1.98f, 0.f);
    glm::vec3 areaNormal = glm::vec3(0.f, -1.f, 0.f);
    glm::vec3 areaColor = glm::vec3(1.f);

    auto sceneIt = loadedScenes.find("helmet");
    if (sceneIt != loadedScenes.end() && sceneIt->second && sceneIt->second->areaLight.valid) {
        const AreaLight& area = sceneIt->second->areaLight;
        areaPos = area.position;
        areaNormal = area.normal;
        areaColor = area.color;
    }

    sceneData.areaLightPosition = glm::vec4(areaPos, areaLightIntensity);
    sceneData.areaLightColor = glm::vec4(areaColor, 1.f);
    sceneData.shadowParams = glm::vec4(20.f, 0.05f, 0.002f, 0.f);
    sceneData.pcssParams = glm::vec4(areaLightWidth, areaLightHeight, pcssMaxPenumbra, 0.05f);

    const float nearPlane = 0.05f;
    const float farPlane = sceneData.shadowParams.x;
    glm::vec3 shadowOrigin = areaPos + areaNormal * 0.02f;
    glm::mat4 shadowProj = glm::perspective(glm::radians(90.f), 1.f, nearPlane, farPlane);
    shadowProj[1][1] *= -1;
    glm::vec3 up = (glm::abs(areaNormal.z) < 0.99f) ? glm::vec3(0.f, 0.f, -1.f) : glm::vec3(1.f, 0.f, 0.f);
    glm::mat4 shadowView = glm::lookAt(shadowOrigin, shadowOrigin + areaNormal, up);
    sceneData.lightViewProj = shadowProj * shadowView;

    auto end = std::chrono::system_clock::now();

    //convert to microseconds (integer), and then come back to miliseconds
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.scene_update_time = elapsed.count() / 1000.f;
}
