#version 450

layout(push_constant) uniform SkyboxPC {
	mat4 view;
	mat4 proj;
} pc;

layout(location = 0) out vec3 outDir;

const vec3 cubeVerts[36] = vec3[](
	vec3(-1,-1,-1), vec3( 1, 1,-1), vec3( 1,-1,-1),
	vec3(-1,-1,-1), vec3(-1, 1,-1), vec3( 1, 1,-1),
	vec3(-1,-1, 1), vec3( 1,-1, 1), vec3( 1, 1, 1),
	vec3(-1,-1, 1), vec3( 1, 1, 1), vec3(-1, 1, 1),
	vec3(-1, 1,-1), vec3(-1, 1, 1), vec3( 1, 1, 1),
	vec3(-1, 1,-1), vec3( 1, 1, 1), vec3( 1, 1,-1),
	vec3(-1,-1,-1), vec3( 1,-1,-1), vec3( 1,-1, 1),
	vec3(-1,-1,-1), vec3( 1,-1, 1), vec3(-1,-1, 1),
	vec3( 1,-1,-1), vec3( 1, 1,-1), vec3( 1, 1, 1),
	vec3( 1,-1,-1), vec3( 1, 1, 1), vec3( 1,-1, 1),
	vec3(-1,-1,-1), vec3(-1,-1, 1), vec3(-1, 1, 1),
	vec3(-1,-1,-1), vec3(-1, 1, 1), vec3(-1, 1,-1)
);

void main()
{
	vec3 pos = cubeVerts[gl_VertexIndex];
	outDir = pos;
	mat4 rotView = mat4(mat3(pc.view));
	vec4 clip = pc.proj * rotView * vec4(pos, 1.0);
	// reverse-Z: z/w = 0 is the far plane
	gl_Position = vec4(clip.xy, 0.0, clip.w);
}
