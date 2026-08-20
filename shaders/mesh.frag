#version 450

#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inWorldPos;
layout (location = 4) in vec4 inTangent;

layout (location = 0) out vec4 outFragColor;

const vec2 poissonDisk[16] = vec2[](
	vec2(-0.94201624, -0.39906216),
	vec2(0.94558609, -0.76890725),
	vec2(-0.094184101, -0.92938870),
	vec2(0.34495938, 0.29387760),
	vec2(-0.91588581, 0.45771432),
	vec2(-0.81544232, -0.87912464),
	vec2(-0.38277543, 0.27676845),
	vec2(0.97484398, 0.75648379),
	vec2(0.44323325, -0.97511554),
	vec2(0.53742981, -0.47373420),
	vec2(-0.26496911, -0.41893023),
	vec2(0.79197514, 0.19090188),
	vec2(-0.24188840, 0.99706507),
	vec2(-0.81409955, 0.91437590),
	vec2(0.19984126, 0.78641367),
	vec2(0.14383161, -0.14100790)
);

float pcf2D(vec2 uv, float dReceiver, float bias, float radius)
{
	float visibility = 0.0;
	for (int i = 0; i < 16; ++i) {
		vec2 sampleUV = uv + poissonDisk[i] * radius;
		float closest = texture(shadowMap, sampleUV).r;
		visibility += (dReceiver - bias > closest) ? 0.0 : 1.0;
	}
	return visibility / 16.0;
}

float areaPcssShadow(vec3 worldPos)
{
	vec4 lightClip = sceneData.lightViewProj * vec4(worldPos, 1.0);
	if (lightClip.w <= 0.0) {
		return 0.0;
	}

	vec3 proj = lightClip.xyz / lightClip.w;
	vec2 uv = proj.xy * 0.5 + 0.5;
	if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z < 0.0 || proj.z > 1.0) {
		return 0.0;
	}

	float dReceiver = proj.z;
	float bias = sceneData.shadowParams.z;
	float lightWidth = sceneData.pcssParams.x;
	float lightHeight = sceneData.pcssParams.y;
	float maxPenumbra = max(sceneData.pcssParams.z, 0.001);
	float dist = length(worldPos - sceneData.areaLightPosition.xyz);
	float lightSizeUV = max(lightWidth, lightHeight) / max(2.0 * dist, 0.001);
	float searchWidth = clamp(lightSizeUV, 0.0, maxPenumbra);

	float blockerSum = 0.0;
	int blockerCount = 0;
	for (int i = 0; i < 16; ++i) {
		vec2 sampleUV = clamp(uv + poissonDisk[i] * searchWidth, 0.0, 1.0);
		float d = texture(shadowMap, sampleUV).r;
		if (dReceiver - bias > d) {
			blockerSum += d;
			blockerCount++;
		}
	}

	if (blockerCount == 0) {
		return 1.0;
	}

	float avgBlocker = blockerSum / float(blockerCount);
	float penumbra = (dReceiver - avgBlocker) * lightSizeUV / max(avgBlocker, 1e-5);
	penumbra = clamp(penumbra, 0.001, maxPenumbra);
	return pcf2D(uv, dReceiver, bias, penumbra);
}

float pointCubeShadow(vec3 lightVec, float dist, float bias, float farPlane)
{
	vec3 dir = normalize(lightVec);
	vec3 up = abs(dir.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangent = normalize(cross(up, dir));
	vec3 bitangent = cross(dir, tangent);
	float filterRadius = 0.02;

	float visibility = 0.0;
	for (int i = 0; i < 16; ++i) {
		vec3 offset = (tangent * poissonDisk[i].x + bitangent * poissonDisk[i].y) * filterRadius;
		float closest = texture(shadowCube, lightVec + offset).r * farPlane;
		visibility += (dist - bias > closest) ? 0.0 : 1.0;
	}
	return visibility / 16.0;
}

const float PI = 3.14159265;

float D_GGX(float NdotH, float a)
{
	float a2 = a * a;
	float d = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
	return a2 / (PI * d * d);
}

float G_SchlickGGX(float NdotX, float k)
{
	return NdotX / (NdotX * (1.0 - k) + k);
}

float G_Smith(float NdotV, float NdotL, float roughness)
{
	float r = roughness + 1.0;
	float k = (r * r) / 8.0;
	return G_SchlickGGX(NdotV, k) * G_SchlickGGX(NdotL, k);
}

vec3 F_Schlick(float VdotH, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

vec3 cookTorrance(vec3 albedo, float metallic, float roughness, vec3 N, vec3 L, vec3 V, vec3 lightColor, float atten)
{
	float NdotL = max(dot(N, L), 0.0);
	if (NdotL <= 0.0) {
		return vec3(0.0);
	}

	vec3 H = normalize(L + V);
	float NdotV = max(dot(N, V), 0.0);
	float NdotH = max(dot(N, H), 0.0);
	float VdotH = max(dot(V, H), 0.0);

	float a = max(roughness * roughness, 0.002);
	vec3 F0 = mix(vec3(0.04), albedo, metallic);
	vec3 F = F_Schlick(VdotH, F0);
	float D = D_GGX(NdotH, a);
	float G = G_Smith(NdotV, NdotL, roughness);

	vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4);
	vec3 diffuse = (1.0 - F) * (1.0 - metallic) * albedo / PI;
	return (diffuse + specular) * lightColor * atten * NdotL;
}

vec3 sampleNormal()
{
	vec3 N = normalize(inNormal);
	vec3 T = inTangent.xyz;
	if (dot(T, T) < 1e-8) {
		return N;
	}
	T = normalize(T - N * dot(N, T));
	vec3 B = normalize(cross(N, T) * inTangent.w);
	vec3 ntex = texture(normalTex, inUV).xyz * 2.0 - 1.0;
	ntex.xy *= materialData.extra0.x;
	return normalize(mat3(T, B, N) * ntex);
}

void main() 
{
	vec3 N = sampleNormal();
	vec3 color = inColor * texture(colorTex, inUV).xyz;
	vec4 mrSample = texture(metalRoughTex, inUV);
	float metallic = materialData.metal_rough_factors.x * mrSample.b;
	float roughness = materialData.metal_rough_factors.y * mrSample.g;
	vec3 ambient = color * sceneData.ambientColor.rgb * sceneData.ambientColor.a;
	vec3 V = normalize(sceneData.cameraPosition.xyz - inWorldPos);
	float farPlane = sceneData.shadowParams.x;
	float NdotV = max(dot(N, V), 0.0);

	vec3 lighting = vec3(0.0);

	{
		vec3 lightVec = inWorldPos - sceneData.pointLightPosition.xyz;
		float dist = length(lightVec);
		vec3 L = -lightVec / max(dist, 0.0001);
		float atten = sceneData.pointLightPosition.w / (dist * dist + 1.0);
		float shadow = 0.0;
		if (dist < farPlane) {
			shadow = pointCubeShadow(lightVec, dist, sceneData.shadowParams.y, farPlane);
		}
		lighting += cookTorrance(color, metallic, roughness, N, L, V, sceneData.pointLightColor.rgb, atten) * shadow;
	}

	{
		vec3 lightVec = inWorldPos - sceneData.areaLightPosition.xyz;
		float dist = length(lightVec);
		vec3 L = -lightVec / max(dist, 0.0001);
		float atten = sceneData.areaLightPosition.w / (dist * dist + 1.0);
		float shadow = 0.0;
		bool belowLight = inWorldPos.y < sceneData.areaLightPosition.y;
		if (dist < farPlane && belowLight) {
			shadow = areaPcssShadow(inWorldPos);
		}
		lighting += cookTorrance(color, metallic, roughness, N, L, V, sceneData.areaLightColor.rgb, atten) * shadow;
	}

	vec3 F0 = mix(vec3(0.04), color, metallic);
	vec3 F = F_Schlick(NdotV, F0);
	vec3 kd = (1.0 - F) * (1.0 - metallic);
	vec3 diffuseIBL = kd * color / PI * texture(irradianceMap, N).rgb;

	vec3 R = reflect(-V, N);
	float lod = roughness * 7.0;
	vec3 prefiltered = textureLod(prefilteredMap, R, lod).rgb;
	vec2 brdf = texture(brdfLUT, vec2(NdotV, roughness)).rg;
	vec3 specularIBL = prefiltered * (F0 * brdf.x + brdf.y);
	lighting += (diffuseIBL + specularIBL) * sceneData.specularParams.w;

	vec3 emissive = materialData.emissiveFactor.rgb * texture(emissiveTex, inUV).rgb;
	outFragColor = vec4(lighting + ambient + emissive, 1.0f);
}
