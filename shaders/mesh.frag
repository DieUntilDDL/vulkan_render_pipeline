#version 450

#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inWorldPos;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	vec3 N = normalize(inNormal);
	vec3 lightVec = inWorldPos - sceneData.pointLightPosition.xyz;
	float dist = length(lightVec);
	vec3 L = -lightVec / max(dist, 0.0001);
	float ndotl = max(dot(N, L), 0.0);
	float atten = sceneData.pointLightPosition.w / (dist * dist + 1.0);

	vec3 color = inColor * texture(colorTex, inUV).xyz;
	vec3 ambient = color * sceneData.ambientColor.xyz;

	float farPlane = sceneData.shadowParams.x;
	float bias = sceneData.shadowParams.y;
	float shadow = 1.0;
	if (dist < farPlane) {
		float closest = texture(shadowCube, lightVec).r * farPlane;
		shadow = (dist - bias > closest) ? 0.0 : 1.0;
	}

	vec3 lighting = color * ndotl * atten * sceneData.pointLightColor.rgb * shadow;
	outFragColor = vec4(lighting + ambient, 1.0f);
}
