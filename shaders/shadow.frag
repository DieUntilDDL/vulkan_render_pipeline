#version 450

layout(set = 0, binding = 0) uniform ShadowData {
	mat4 lightViewProj;
	vec4 lightPosFar;
} shadowData;

layout(location = 0) in vec3 inWorldPos;

void main()
{
	float dist = length(inWorldPos - shadowData.lightPosFar.xyz);
	gl_FragDepth = clamp(dist / shadowData.lightPosFar.w, 0.0, 1.0);
}
