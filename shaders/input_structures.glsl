layout(set = 0, binding = 0) uniform SceneData{   

	mat4 view;
	mat4 proj;
	mat4 viewproj;
	mat4 lightViewProj;
	vec4 ambientColor; // rgb = color, a = intensity
	vec4 pointLightPosition; // xyz = point light, w = intensity
	vec4 pointLightColor;
	vec4 areaLightPosition; // xyz = area-light center, w = intensity
	vec4 areaLightColor;
	vec4 shadowParams; // x = far, y = point bias, z = area projective bias
	vec4 cameraPosition; // xyz = world position
	vec4 specularParams; // x = ks, y = shininess, z = roughness influence, w = IBL intensity
	vec4 pcssParams; // x = light width, y = light height, z = max penumbra UV, w = near
} sceneData;

layout(set = 0, binding = 1) uniform sampler2D shadowMap;
layout(set = 0, binding = 2) uniform samplerCube shadowCube;
layout(set = 0, binding = 3) uniform samplerCube irradianceMap;
layout(set = 0, binding = 4) uniform samplerCube prefilteredMap;
layout(set = 0, binding = 5) uniform sampler2D brdfLUT;

layout(set = 1, binding = 0) uniform GLTFMaterialData{   

	vec4 colorFactors;
	vec4 metal_rough_factors;
	vec4 emissiveFactor;
	vec4 extra0; // x = normal scale
	
} materialData;

layout(set = 1, binding = 1) uniform sampler2D colorTex;
layout(set = 1, binding = 2) uniform sampler2D metalRoughTex;
layout(set = 1, binding = 3) uniform sampler2D emissiveTex;
layout(set = 1, binding = 4) uniform sampler2D normalTex;
