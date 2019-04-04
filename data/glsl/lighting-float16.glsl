#extension GL_EXT_shader_explicit_arithmetic_types_float16: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8: enable

struct Light
{
   vec4 pos;
   f16vec4 diffuse;
   f16vec4 ambient;
   f16vec4 specular;
   f16vec4 attenuation;
   f16vec4 rotation;
   f16vec4 direction;
   f16vec4 spot_angle_attenuation;
   float16_t spot_cutoff;
   float16_t spot_cutoff_angle;
   float16_t spot_padding_0;
   float16_t spot_padding_1;
   mat4 view_matrix;
   mat4 view_matrix_inv;
   float16_t intensity;
   bool casts_shadows;
   float16_t volume_scale_cap;
   float16_t volume_cutoff;
   uint dirty;
   uint cached;
   float pad0;
   float pad1;
};

struct Material
{
   f16vec4 diffuse;
   f16vec4 ambient;
   f16vec4 specular;
   float16_t shininess;
   uint8_t diffuse_tex_count;
   uint8_t normal_tex_count;
   uint8_t specular_tex_count;
   uint8_t opacity_tex_count;
   float16_t reflectiveness;
   float16_t roughness;
   float16_t emission;
};

struct LightColor
{
   f16vec3 diffuse;
   f16vec3 ambient;
   f16vec3 specular;
};

float16_t
compute_spotlight_cutoff_factor(Light l, f16vec3 light_to_pos_norm)
{
   // Compute angle of this light beam with the spotlight's direction
   f16vec3 spotlight_dir_norm = normalize(f16vec3(l.direction));
   float16_t dp_angle_with_light = dot(light_to_pos_norm, spotlight_dir_norm);

   float16_t cutoff_factor = 0.0hf;
   if (dp_angle_with_light >= l.spot_cutoff) {
      /* Beam is inside the light cone, attenuate with angular distance
       * to the center of the beam
       */
      float16_t dist = 90.0hf * (1.0hf - (dp_angle_with_light - l.spot_cutoff) / (1.0hf - l.spot_cutoff));
      f16vec3 att = l.spot_angle_attenuation.xyz;
      cutoff_factor = 1.0hf / (att.x + att.y * dist + att.z * dist * dist);
   }

   return cutoff_factor;
}

float16_t
compute_shadow_factor(vec4 light_space_pos,
                      sampler2DShadow shadow_map,
                      uint shadow_map_size,
                      uint pcf_size)
{
   // Convert light space position to NDC
   vec3 light_space_ndc = light_space_pos.xyz /= light_space_pos.w;

   // If the fragment is outside the light's projection then it is outside
   // the light's influence, which means it is in the shadow (notice that
   // such fragment position is outside the shadow map texture so it would
   // it be incorrect to sample the shadow map with it)
   if (abs(light_space_ndc.x) > 1.0 ||
       abs(light_space_ndc.y) > 1.0 ||
       light_space_ndc.z > 1.0)
      return 0.0hf;

   // Translate from NDC to shadow map space (Vulkan's Z is already in [0..1])
   vec2 shadow_map_coord = light_space_ndc.xy * 0.5 + 0.5;

   // compute total number of samples to take from the shadow map
   int pcf_size_minus_1 = int(pcf_size - 1);
   float kernel_size = 2.0 * pcf_size_minus_1 + 1.0;
   float num_samples = kernel_size * kernel_size;

   // Counter for the shadow map samples not in the shadow
   float lighted_count = 0.0;

   // Take samples from the shadow map
   float shadow_map_texel_size = 1.0 / shadow_map_size;
   for (int x = -pcf_size_minus_1; x <= pcf_size_minus_1; x++)
   for (int y = -pcf_size_minus_1; y <= pcf_size_minus_1; y++) {
      // Compute coordinate for this PCF sample
      vec3 pcf_coord =
		vec3(shadow_map_coord + vec2(x, y) * shadow_map_texel_size,
             light_space_ndc.z);

      // Check if the sample is in light or in the shadow
      lighted_count += texture(shadow_map, pcf_coord);

   }

   return float16_t(lighted_count / num_samples);
}

/*
LightColor
compute_lighting(Light l,
                 vec3 world_pos,
                 vec3 normal,
                 vec3 view_dir,
                 Material mat,
                 bool receives_shadows,
                 vec4 light_space_pos,
                 sampler2DShadow shadow_map,
                 uint shadow_map_size,
                 uint pcf_size)
{
   f16vec3 light_to_pos_norm;
   float16_t att_factor;
   float16_t cutoff_factor;

   // Check if the fragment is in the shadow
   float16_t shadow_factor;
   if (receives_shadows) {
      shadow_factor = compute_shadow_factor(light_space_pos, shadow_map,
                                            shadow_map_size, pcf_size);
   } else {
      shadow_factor = 1.0hf;
   }

   if (l.pos.w == 0.0) {
      // Directional light, no attenuation, no cutoff
      light_to_pos_norm = f16vec3(normalize(vec3(l.pos)));
      att_factor = 1.0hf;
      cutoff_factor = 1.0hf;
   } else {
      // Positional light, compute attenuation
      vec3 light_to_pos = world_pos - l.pos.xyz;
      light_to_pos_norm = f16vec3(normalize(light_to_pos));
      float16_t dist = float16_t(length(light_to_pos));
      att_factor = 1.0hf / (l.attenuation.x + l.attenuation.y * dist +
                          l.attenuation.z * dist * dist);

      if (l.pos.w == 1.0) {
         // Point light: no cutoff, normal ambient attenuation
         cutoff_factor = 1.0hf;
      } else {
         cutoff_factor = compute_spotlight_cutoff_factor(l, light_to_pos_norm);
      }
   }

   // Compute reflection from light for this fragment
   normal = normalize(normal);
   f16vec3 norm = f16vec3(normalize(normal));
   float16_t dp_reflection = max(0.0hf, dot(norm, -light_to_pos_norm));

   shadow_factor *= cutoff_factor;

   // Compute light contributions to the fragment.
   LightColor lc;
   lc.diffuse = mat.diffuse.xyz * l.diffuse.xyz * att_factor *
                dp_reflection * shadow_factor * l.intensity;
   lc.ambient = mat.ambient.xyz * l.ambient.xyz * att_factor * l.intensity;

   lc.specular = f16vec3(0.0hf);
   if (dot(norm, -light_to_pos_norm) >= 0.0) {
      f16vec3 reflection_dir = reflect(light_to_pos_norm, norm);
      float16_t shine_factor = dot(reflection_dir, f16vec3(normalize(view_dir)));
      lc.specular =
           att_factor * l.specular.xyz * mat.specular.xyz *
           pow(max(0.0hf, shine_factor), mat.shininess) * shadow_factor *
           l.intensity;
   }

   return lc;
}
LightColor
compute_lighting(Light l,
                 vec3 world_pos,
                 vec3 normal,
                 vec3 view_dir,
                 Material mat)
{
   vec3 light_to_pos_norm;
   float att_factor;
   float cutoff_factor;

   if (l.pos.w == 0.0) {
      // Directional light, no attenuation, no cutoff
      light_to_pos_norm = normalize(vec3(l.pos));
      att_factor = 1.0;
      cutoff_factor = 1.0;
   } else {
      // Positional light, compute attenuation
      vec3 light_to_pos = world_pos - l.pos.xyz;
      light_to_pos_norm = normalize(light_to_pos);
      float dist = length(light_to_pos);
      att_factor = 1.0 / (l.attenuation.x + l.attenuation.y * dist +
                          l.attenuation.z * dist * dist);

      if (l.pos.w == 1.0) {
         // Point light: no cutoff, normal ambient attenuation
         cutoff_factor = 1.0f;
      } else {
         cutoff_factor = compute_spotlight_cutoff_factor(l, light_to_pos_norm);
      }
   }

   // Compute reflection from light for this fragment
   normal = normalize(normal);
   float dp_reflection = max(0.0, dot(normal, -light_to_pos_norm));

   // No shadowing
   float shadow_factor = cutoff_factor;

   // Compute light contributions to the fragment.
   LightColor lc;
   lc.diffuse = mat.diffuse.xyz * l.diffuse.xyz * att_factor *
                dp_reflection * shadow_factor * l.intensity;
   lc.ambient = mat.ambient.xyz * l.ambient.xyz * att_factor * l.intensity;

   lc.specular = vec3(0);
   if (dot(normal, -light_to_pos_norm) >= 0.0) {
      vec3 reflection_dir = reflect(light_to_pos_norm, normal);
      float shine_factor = dot(reflection_dir, normalize(view_dir));
      lc.specular =
           att_factor * l.specular.xyz * mat.specular.xyz *
           pow(max(0.0, shine_factor), mat.shininess) * shadow_factor *
           l.intensity;
   }

   return lc;
}

LightColor
compute_lighting_point(Light l,
                       vec3 world_pos,
                       vec3 normal,
                       vec3 view_dir,
                       Material mat)
{
   // Compute attenuation
   vec3 light_to_pos = world_pos - l.pos.xyz;
   vec3 light_to_pos_norm = normalize(light_to_pos);
   float dist = length(light_to_pos);
   float att_factor = l.intensity /
                      (l.attenuation.x + l.attenuation.y * dist +
                       l.attenuation.z * dist * dist);

   // Compute reflection from light for this fragment
   normal = normalize(normal);
   float dp_reflection = max(0.0, dot(normal, -light_to_pos_norm));

   // Compute light contributions to the fragment.
   LightColor lc;
   lc.diffuse = mat.diffuse.xyz * l.diffuse.xyz * att_factor *
                dp_reflection;
   lc.ambient = mat.ambient.xyz * l.ambient.xyz * att_factor;

   lc.specular = vec3(0);
   if (dot(normal, -light_to_pos_norm) >= 0.0) {
      vec3 reflection_dir = reflect(light_to_pos_norm, normal);
      float shine_factor = dot(reflection_dir, normalize(view_dir));
      lc.specular =
           l.specular.xyz * mat.specular.xyz *
           pow(max(0.0, shine_factor), mat.shininess) * att_factor;
   }

   return lc;
}

LightColor
compute_lighting_point_diffuse(Light l,
                               vec3 world_pos,
                               vec3 normal,
                               Material mat)
{
   // Compute attenuation
   vec3 light_to_pos = world_pos - l.pos.xyz;
   vec3 light_to_pos_norm = normalize(light_to_pos);
   float dist = length(light_to_pos);
   float att_factor = l.intensity /
                      (l.attenuation.x + l.attenuation.y * dist +
                       l.attenuation.z * dist * dist);

   // Compute reflection from light for this fragment
   normal = normalize(normal);
   float dp_reflection = max(0.0, dot(normal, -light_to_pos_norm));

   // Compute light contributions to the fragment.
   LightColor lc;
   lc.diffuse = mat.diffuse.xyz * l.diffuse.xyz * att_factor *
                dp_reflection;

   lc.ambient = lc.specular = vec3(0.0);

   return lc;
}

LightColor
compute_lighting_point_diffuse_specular(Light l,
                                        vec3 world_pos,
                                        vec3 normal,
                                        vec3 view_dir,
                                        Material mat)
{
   // Compute attenuation
   vec3 light_to_pos = world_pos - l.pos.xyz;
   vec3 light_to_pos_norm = normalize(light_to_pos);
   float dist = length(light_to_pos);
   float att_factor = l.intensity /
                      (l.attenuation.x + l.attenuation.y * dist +
                       l.attenuation.z * dist * dist);

   // Compute reflection from light for this fragment
   normal = normalize(normal);
   float dp_reflection = max(0.0, dot(normal, -light_to_pos_norm));

   // Compute light contributions to the fragment.
   LightColor lc;
   lc.diffuse = mat.diffuse.xyz * l.diffuse.xyz * att_factor *
                dp_reflection;

   lc.ambient = vec3(0.0);

   lc.specular = vec3(0.0);
   if (dot(normal, -light_to_pos_norm) >= 0.0) {
      vec3 reflection_dir = reflect(light_to_pos_norm, normal);
      float shine_factor = dot(reflection_dir, normalize(view_dir));
      lc.specular =
           l.specular.xyz * mat.specular.xyz *
           pow(max(0.0, shine_factor), mat.shininess) * att_factor;
   }

   return lc;
}

LightColor
compute_lighting_spot(Light l,
                      vec3 world_pos,
                      vec3 normal,
                      vec3 view_dir,
                      Material mat)
{
   // Compute attenuation
   vec3 light_to_pos = world_pos - l.pos.xyz;
   vec3 light_to_pos_norm = normalize(light_to_pos);
   float dist = length(light_to_pos);
   float att_factor = l.intensity /
                      (l.attenuation.x + l.attenuation.y * dist +
                       l.attenuation.z * dist * dist);

   // Compute spotlight cutoff
   float cutoff_factor = compute_spotlight_cutoff_factor(l, light_to_pos_norm);
   att_factor *= cutoff_factor;

   // Compute reflection from light for this fragment
   normal = normalize(normal);
   float dp_reflection = max(0.0, dot(normal, -light_to_pos_norm));

   // Compute light contributions to the fragment.
   LightColor lc;
   lc.diffuse = mat.diffuse.xyz * l.diffuse.xyz * dp_reflection * att_factor;
   lc.ambient = mat.ambient.xyz * l.ambient.xyz * att_factor;

   lc.specular = vec3(0);
   if (dot(normal, -light_to_pos_norm) >= 0.0) {
      vec3 reflection_dir = reflect(light_to_pos_norm, normal);
      float shine_factor = dot(reflection_dir, normalize(view_dir));
      lc.specular =
           l.specular.xyz * mat.specular.xyz *
           pow(max(0.0, shine_factor), mat.shininess) * att_factor;
   }

   return lc;
}

LightColor
compute_lighting_spot_diffuse(Light l,
                               vec3 world_pos,
                               vec3 normal,
                               Material mat)
{
   // Compute attenuation
   vec3 light_to_pos = world_pos - l.pos.xyz;
   vec3 light_to_pos_norm = normalize(light_to_pos);
   float dist = length(light_to_pos);
   float att_factor = l.intensity /
                      (l.attenuation.x + l.attenuation.y * dist +
                       l.attenuation.z * dist * dist);

   // Compute spotlight cutoff
   float cutoff_factor = compute_spotlight_cutoff_factor(l, light_to_pos_norm);
   att_factor *= cutoff_factor;

   // Compute reflection from light for this fragment
   normal = normalize(normal);
   float dp_reflection = max(0.0, dot(normal, -light_to_pos_norm));

   // Compute light contributions to the fragment.
   LightColor lc;
   lc.diffuse = mat.diffuse.xyz * l.diffuse.xyz * dp_reflection * att_factor;

   lc.ambient = lc.specular = vec3(0);

   return lc;
}

LightColor
compute_lighting_spot_diffuse_specular(Light l,
                                       vec3 world_pos,
                                       vec3 normal,
                                       vec3 view_dir,
                                       Material mat)
{
   // Compute attenuation
   vec3 light_to_pos = world_pos - l.pos.xyz;
   vec3 light_to_pos_norm = normalize(light_to_pos);
   float dist = length(light_to_pos);
   float att_factor = l.intensity /
                      (l.attenuation.x + l.attenuation.y * dist +
                       l.attenuation.z * dist * dist);

   // Compute spotlight cutoff
   float cutoff_factor = compute_spotlight_cutoff_factor(l, light_to_pos_norm);
   att_factor *= cutoff_factor;

   // Compute reflection from light for this fragment
   normal = normalize(normal);
   float dp_reflection = max(0.0, dot(normal, -light_to_pos_norm));

   // Compute light contributions to the fragment.
   LightColor lc;
   lc.diffuse = mat.diffuse.xyz * l.diffuse.xyz * dp_reflection * att_factor;

   lc.ambient = vec3(0);

   lc.specular = vec3(0);
   if (dot(normal, -light_to_pos_norm) >= 0.0) {
      vec3 reflection_dir = reflect(light_to_pos_norm, normal);
      float shine_factor = dot(reflection_dir, normalize(view_dir));
      lc.specular =
           l.specular.xyz * mat.specular.xyz *
           pow(max(0.0, shine_factor), mat.shininess) * att_factor;
   }

   return lc;
}

LightColor
compute_lighting_spot(Light l,
                      vec3 world_pos,
                      vec3 normal,
                      vec3 view_dir,
                      Material mat,
                      bool receives_shadows,
                      vec4 light_space_pos,
                      sampler2DShadow shadow_map,
                      uint shadow_map_size,
                      uint pcf_size)
{
   // Compute attenuation
   vec3 light_to_pos = world_pos - l.pos.xyz;
   vec3 light_to_pos_norm = normalize(light_to_pos);
   float dist = length(light_to_pos);
   float att_factor = l.intensity /
                      (l.attenuation.x + l.attenuation.y * dist +
                       l.attenuation.z * dist * dist);

   // Apply spotlight cutoff
   att_factor *= compute_spotlight_cutoff_factor(l, light_to_pos_norm);

   // Check if the fragment is in the shadow
   float shadow_factor;
   if (receives_shadows) {
      shadow_factor = compute_shadow_factor(light_space_pos, shadow_map,
                                            shadow_map_size, pcf_size);
   } else {
      shadow_factor = 1.0;
   }

   // Compute reflection from light for this fragment
   normal = normalize(normal);
   float dp_reflection = max(0.0, dot(normal, -light_to_pos_norm));

   // Compute light contributions to the fragment.
   LightColor lc;
   lc.diffuse = mat.diffuse.xyz * l.diffuse.xyz * dp_reflection * att_factor *
                shadow_factor;
   lc.ambient = mat.ambient.xyz * l.ambient.xyz * att_factor;

   lc.specular = vec3(0);
   if (dot(normal, -light_to_pos_norm) >= 0.0) {
      vec3 reflection_dir = reflect(light_to_pos_norm, normal);
      float shine_factor = dot(reflection_dir, normalize(view_dir));
      lc.specular =
           l.specular.xyz * mat.specular.xyz *
           pow(max(0.0, shine_factor), mat.shininess) *
           att_factor * shadow_factor;
   }

   return lc;
}

LightColor
compute_lighting_directional(Light l,
                             vec3 world_pos,
                             vec3 normal,
                             vec3 view_dir,
                             Material mat)
{
   // Compute reflection from light for this fragment
   vec3 light_to_pos_norm = normalize(vec3(l.pos));
   normal = normalize(normal);
   float dp_reflection = max(0.0, dot(normal, -light_to_pos_norm));

   // Compute light contributions to the fragment.
   LightColor lc;
   lc.diffuse = mat.diffuse.xyz * l.diffuse.xyz *
                dp_reflection * l.intensity;
   lc.ambient = mat.ambient.xyz * l.ambient.xyz * l.intensity;

   lc.specular = vec3(0);
   if (dot(normal, -light_to_pos_norm) >= 0.0) {
      vec3 reflection_dir = reflect(light_to_pos_norm, normal);
      float shine_factor = dot(reflection_dir, normalize(view_dir));
      lc.specular =
           l.specular.xyz * mat.specular.xyz *
           pow(max(0.0, shine_factor), mat.shininess) * l.intensity;
   }

   return lc;
}
*/

LightColor
compute_lighting_directional(Light l,
                             vec3 world_pos,
                             vec3 normal,
                             vec3 view_dir,
                             Material mat,
                             bool receives_shadows,
                             vec4 light_space_pos,
                             sampler2DShadow shadow_map,
                             uint shadow_map_size,
                             uint pcf_size)
{
   // Compute reflection from light for this fragment
   f16vec3 light_to_pos_norm = f16vec3(normalize(vec3(l.pos)));
   normal = normalize(normal);
   float16_t dp_reflection = max(0.0hf, dot(f16vec3(normal), -light_to_pos_norm));

   // No attenuation
   float16_t att_factor = l.intensity;

    // Check if the fragment is in the shadow
   float16_t shadow_factor = compute_shadow_factor(light_space_pos, shadow_map,
                                            shadow_map_size, pcf_size);

  // Compute light contributions to the fragment.
   LightColor lc;
   lc.diffuse = mat.diffuse.xyz * l.diffuse.xyz * dp_reflection * att_factor *
                shadow_factor;
   lc.ambient = mat.ambient.xyz * l.ambient.xyz * att_factor;

   lc.specular = f16vec3(0.0hf);
   if (dot(f16vec3(normal), -light_to_pos_norm) >= 0.0hf) {
      f16vec3 reflection_dir = reflect(light_to_pos_norm, f16vec3(normal));
      float16_t shine_factor = dot(reflection_dir, f16vec3(normalize(view_dir)));
      lc.specular =
           l.specular.xyz * mat.specular.xyz *
           pow(max(0.0hf, shine_factor), mat.shininess) *
           att_factor * shadow_factor;
   }

   return lc;
}
