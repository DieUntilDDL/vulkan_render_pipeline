layout(set = 0, binding = 0) uniform SceneData{   

	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 ambientColor;
	vec4 pointLightPosition; // xyz = world position, w = intensity
	vec4 pointLightColor;
	vec4 shadowParams; // x = far plane, y = bias
} sceneData;

layout(set = 0, binding = 1) uniform samplerCube shadowCube;

layout(set = 1, binding = 0) uniform GLTFMaterialData{   

	vec4 colorFactors;
	vec4 metal_rough_factors;
	
} materialData;

layout(set = 1, binding = 1) uniform sampler2D colorTex;
layout(set = 1, binding = 2) uniform sampler2D metalRoughTex;
