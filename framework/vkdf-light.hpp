#ifndef __VKDF_LIGHT_H__
#define __VKDF_LIGHT_H__

#include "vkdf-deps.hpp"
#include "vkdf-util.hpp"
#include "vkdf-types.hpp"

enum {
   VKDF_LIGHT_DIRECTIONAL = 0,
   VKDF_LIGHT_POINT       = 1,
   VKDF_LIGHT_SPOTLIGHT   = 2,
   VKDF_LIGHT_AMBIENT     = 3
};

enum {
   VKDF_LIGHT_DIRTY            = (1 << 0),
   VKDF_LIGHT_DIRTY_SHADOWS    = (1 << 1),
   VKDF_LIGHT_DIRTY_VIEW       = (1 << 2),
};

enum {
   VKDF_LIGHT_CACHED_VIEW       = (1 << 0),
   VKDF_LIGHT_CACHED_VIEW_INV   = (1 << 1),
};

/* WARNING: changes to this struct must be replicated in lighting.glsl */
typedef struct {
   // Common light attributes
   glm::vec4 origin;      // .w = light type
   glm::vec4 diffuse;
   glm::vec4 ambient;
   glm::vec4 specular;
   glm::vec4 attenuation; // .x = constant, .y = linear, .z = quadratic

   // Spotlights
   struct {
      struct {
         glm::vec4 rot;
         glm::vec4 dir;            // Computed from rotation
      } priv;
      glm::vec4 angle_attenuation; // .x = constant, .y = linear, .z = quadratic
      float cutoff;                // cosine of the spotlight's cutoff angle (half of apeture angle)
      float cutoff_angle;          // spotlight's cutoff angle (half of aperture angle)
      float padding[2];            // Keep this struct 16-byte aligned
   } spot;

   glm::mat4 view_matrix;       // View matrix for the light
   glm::mat4 view_matrix_inv;

   float intensity;             // From 0 (no light) to 1 (full intensity)
   uint32_t casts_shadows;

   float volume_scale_cap;      // Maximum scale of the light volume (for light volume lighting)
   float volume_cutoff;         // Percentage of light being cutoff by the volume (0..1]

   uint32_t dirty;              // Dirty state
   uint32_t cached;
   float padding[2];            // Keep this struct 16-byte aligned
} VkdfLight;

typedef struct {
   // Common light attributes
   glm::vec4 origin;      // .w = light type
   f16vec4 diffuse;
   f16vec4 ambient;
   f16vec4 specular;
   f16vec4 attenuation; // .x = constant, .y = linear, .z = quadratic

   // Spotlights
   struct {
      struct {
         f16vec4 rot;
         f16vec4 dir;            // Computed from rotation
      } priv;
      f16vec4 angle_attenuation; // .x = constant, .y = linear, .z = quadratic
      float16_t cutoff;          // cosine of the spotlight's cutoff angle (half of apeture angle)
      float16_t cutoff_angle;    // spotlight's cutoff angle (half of aperture angle)
      float padding;          // Keep this struct 16-byte aligned
   } spot;

   glm::mat4 view_matrix;       // View matrix for the light
   glm::mat4 view_matrix_inv;

   float16_t intensity;         // From 0 (no light) to 1 (full intensity)

   float16_t volume_scale_cap;      // Maximum scale of the light volume (for light volume lighting)
   float16_t volume_cutoff;         // Percentage of light being cutoff by the volume (0..1]
   float16_t pad1;

   uint32_t casts_shadows;
   uint32_t dirty;              // Dirty state
   uint32_t cached;
   float padding[3];            // Keep this struct 16-byte aligned
} VkdfLight16;



VkdfLight *
vkdf_light_new_directional(glm::vec4 dir,
                           glm::vec4 diffuse,
                           glm::vec4 ambient,
                           glm::vec4 specular);

VkdfLight *
vkdf_light_new_positional(glm::vec4 pos,
                          glm::vec4 diffuse,
                          glm::vec4 ambient,
                          glm::vec4 specular,
                          glm::vec4 attenuation);

VkdfLight *
vkdf_light_new_spotlight(glm::vec4 pos,
                         float cutoff_angle,
                         glm::vec4 diffuse,
                         glm::vec4 ambient,
                         glm::vec4 specular,
                         glm::vec4 attenuation,
                         glm::vec4 angle_attenuation);

VkdfLight *
vkdf_light_new_ambient(glm::vec4 ambient);

void inline
vkdf_light_set_type(VkdfLight *l, uint32_t light_type)
{
   l->origin.w = (float) light_type;

   bitfield_set(&l->dirty, ~0);
   bitfield_unset(&l->cached, ~0);
}

uint32_t inline
vkdf_light_get_type(VkdfLight *l)
{
   return (uint32_t) l->origin.w;
}

void inline
vkdf_light_set_position(VkdfLight *l, glm::vec3 pos)
{
   assert(vkdf_light_get_type(l) != VKDF_LIGHT_DIRECTIONAL &&
          vkdf_light_get_type(l) != VKDF_LIGHT_AMBIENT);
   l->origin = vec4(pos, l->origin.w);

   uint32_t dirty_bits =
      VKDF_LIGHT_DIRTY | VKDF_LIGHT_DIRTY_SHADOWS | VKDF_LIGHT_DIRTY_VIEW;
   bitfield_set(&l->dirty, dirty_bits);

   bitfield_unset(&l->cached,
                  VKDF_LIGHT_CACHED_VIEW | VKDF_LIGHT_CACHED_VIEW_INV);
}

glm::vec4 inline
vkdf_light_get_position_and_type(VkdfLight *l)
{
   assert(vkdf_light_get_type(l) != VKDF_LIGHT_DIRECTIONAL &&
          vkdf_light_get_type(l) != VKDF_LIGHT_AMBIENT);
   return l->origin;
}

glm::vec3 inline
vkdf_light_get_position(VkdfLight *l)
{
   assert(vkdf_light_get_type(l) != VKDF_LIGHT_DIRECTIONAL &&
          vkdf_light_get_type(l) != VKDF_LIGHT_AMBIENT);
   return vec3(l->origin);
}

void inline
vkdf_light_set_direction(VkdfLight *l, glm::vec3 dir)
{
   assert(vkdf_light_get_type(l) == VKDF_LIGHT_DIRECTIONAL &&
          vkdf_light_get_type(l) != VKDF_LIGHT_AMBIENT);
   l->origin = vec4(dir, l->origin.w);

   uint32_t dirty_bits =
      VKDF_LIGHT_DIRTY | VKDF_LIGHT_DIRTY_SHADOWS | VKDF_LIGHT_DIRTY_VIEW;
   bitfield_set(&l->dirty, dirty_bits);

   bitfield_unset(&l->cached,
                  VKDF_LIGHT_CACHED_VIEW | VKDF_LIGHT_CACHED_VIEW_INV);
}

glm::vec3 inline
vkdf_light_get_direction(VkdfLight *l)
{
   uint32_t light_type = vkdf_light_get_type(l);
   if (light_type == VKDF_LIGHT_DIRECTIONAL)
      return vec3(l->origin);
   else if (light_type == VKDF_LIGHT_SPOTLIGHT)
      return vec3(l->spot.priv.dir);
   assert(!"Light type does not have a direction vector");
}

void inline
vkdf_light_set_diffuse(VkdfLight *l, glm::vec4 color)
{
   assert(vkdf_light_get_type(l) != VKDF_LIGHT_AMBIENT);

   l->diffuse = color;

   uint32_t dirty_bits = VKDF_LIGHT_DIRTY;
   bitfield_set(&l->dirty, dirty_bits);
}

glm::vec4 inline
vkdf_light_get_diffuse(VkdfLight *l)
{
   assert(vkdf_light_get_type(l) != VKDF_LIGHT_AMBIENT);
   return l->diffuse;
}

void inline
vkdf_light_set_ambient(VkdfLight *l, glm::vec4 color)
{
   l->ambient = color;

   uint32_t dirty_bits = VKDF_LIGHT_DIRTY;
   bitfield_set(&l->dirty, dirty_bits);
}

glm::vec4 inline
vkdf_light_get_ambient(VkdfLight *l)
{
   return l->ambient;
}

void inline
vkdf_light_set_specular(VkdfLight *l, glm::vec4 color)
{
   assert(vkdf_light_get_type(l) != VKDF_LIGHT_AMBIENT);

   l->specular = color;

   uint32_t dirty_bits = VKDF_LIGHT_DIRTY;
   bitfield_set(&l->dirty, dirty_bits);
}

glm::vec4 inline
vkdf_light_get_specular(VkdfLight *l)
{
   assert(vkdf_light_get_type(l) != VKDF_LIGHT_AMBIENT);
   return l->specular;
}

void inline
vkdf_light_set_attenuation(VkdfLight *l, glm::vec4 attenuation)
{
   assert(vkdf_light_get_type(l) != VKDF_LIGHT_AMBIENT);

   l->attenuation = attenuation;

   uint32_t dirty_bits = VKDF_LIGHT_DIRTY;
   bitfield_set(&l->dirty, dirty_bits);
}

glm::vec4 inline
vkdf_light_get_attenuation(VkdfLight *l)
{
   assert(vkdf_light_get_type(l) != VKDF_LIGHT_AMBIENT);
   return l->attenuation;
}

void inline
vkdf_light_set_angle_attenuation(VkdfLight *l, glm::vec4 attenuation)
{
   assert(vkdf_light_get_type(l) == VKDF_LIGHT_SPOTLIGHT);

   l->spot.angle_attenuation = attenuation;

   uint32_t dirty_bits = VKDF_LIGHT_DIRTY;
   bitfield_set(&l->dirty, dirty_bits);
}

glm::vec4 inline
vkdf_light_get_angle_attenuation(VkdfLight *l)
{
   assert(vkdf_light_get_type(l) == VKDF_LIGHT_SPOTLIGHT);

   return l->spot.angle_attenuation;
}

void inline
vkdf_light_set_cutoff_angle(VkdfLight *l, float angle)
{
   assert(vkdf_light_get_type(l) == VKDF_LIGHT_SPOTLIGHT);
   l->spot.cutoff_angle = angle;
   l->spot.cutoff = cosf(l->spot.cutoff_angle);

   uint32_t dirty_bits =
      VKDF_LIGHT_DIRTY | VKDF_LIGHT_DIRTY_SHADOWS | VKDF_LIGHT_DIRTY_VIEW;
   bitfield_set(&l->dirty, dirty_bits);

   bitfield_unset(&l->cached,
                  VKDF_LIGHT_CACHED_VIEW | VKDF_LIGHT_CACHED_VIEW_INV);
}

/* The cutoff angle is half of the aperture angle of the spotlight */
void inline
vkdf_light_set_aperture_angle(VkdfLight *l, float angle)
{
   vkdf_light_set_cutoff_angle(l, angle / 2.0f);
}

float inline
vkdf_light_get_cutoff_angle(VkdfLight *l)
{
   assert(vkdf_light_get_type(l) == VKDF_LIGHT_SPOTLIGHT);
   return l->spot.cutoff_angle;
}

float inline
vkdf_light_get_aperture_angle(VkdfLight *l)
{
   assert(vkdf_light_get_type(l) == VKDF_LIGHT_SPOTLIGHT);
   return 2.0f * l->spot.cutoff_angle;
}

float inline
vkdf_light_get_cutoff_factor(VkdfLight *l)
{
   assert(vkdf_light_get_type(l) == VKDF_LIGHT_SPOTLIGHT);
   return l->spot.cutoff;
}

// WARNING: do not write light.{rot,dir} directly from applications,
// always do it through this function so we update both at once
void inline
vkdf_light_set_rotation(VkdfLight *l, glm::vec3 rot)
{
   assert(vkdf_light_get_type(l) == VKDF_LIGHT_SPOTLIGHT);
   l->spot.priv.rot = glm::vec4(rot, 0.0f);
   l->spot.priv.dir =
      glm::vec4(vkdf_compute_viewdir(glm::vec3(l->spot.priv.rot)), 0.0f);

   uint32_t dirty_bits =
      VKDF_LIGHT_DIRTY | VKDF_LIGHT_DIRTY_SHADOWS | VKDF_LIGHT_DIRTY_VIEW;

   bitfield_set(&l->dirty, dirty_bits);

   bitfield_unset(&l->cached,
                  VKDF_LIGHT_CACHED_VIEW | VKDF_LIGHT_CACHED_VIEW_INV);
}

glm::vec3 inline
vkdf_light_get_rotation(VkdfLight *l)
{
   assert(vkdf_light_get_type(l) == VKDF_LIGHT_SPOTLIGHT);
   return l->spot.priv.rot;
}

void inline
vkdf_light_enable_shadows(VkdfLight *l, bool enable)
{
   l->casts_shadows = (uint32_t) enable;

   uint32_t dirty_bits = VKDF_LIGHT_DIRTY | VKDF_LIGHT_DIRTY_SHADOWS;
   bitfield_set(&l->dirty, dirty_bits);
}

bool inline
vkdf_light_casts_shadows(VkdfLight *l)
{
   return (bool) l->casts_shadows;
}

void inline
vkdf_light_look_at(VkdfLight *l, glm::vec3 target)
{
   glm::vec3 rot = vkdf_compute_view_rotation(glm::vec3(l->origin), target);
   vkdf_light_set_rotation(l, rot);
}

const glm::mat4 *
vkdf_light_get_view_matrix(VkdfLight *l);

const glm::mat4 *
vkdf_light_get_view_matrix_inv(VkdfLight *l);

void inline
vkdf_light_set_dirty(VkdfLight *l, bool dirty)
{
   if (!dirty)
      l->dirty = 0;
   else
      bitfield_set(&l->dirty, VKDF_LIGHT_DIRTY);
}

void inline
vkdf_light_set_dirty_shadows(VkdfLight *l, bool dirty)
{
   if (dirty)
      bitfield_set(&l->dirty, VKDF_LIGHT_DIRTY_SHADOWS);
   else
      bitfield_unset(&l->dirty, VKDF_LIGHT_DIRTY_SHADOWS);

   /* If we the shadow map data is dirty then the light is too, but
    * otherwise we want to keep the dirty flag
    */
   if (dirty)
      bitfield_set(&l->dirty, VKDF_LIGHT_DIRTY);
}

bool inline
vkdf_light_is_dirty(VkdfLight *l)
{
   return (bool) l->dirty;
}

bool inline
vkdf_light_has_dirty_shadows(VkdfLight *l)
{
   return l->casts_shadows && bitfield_get(l->dirty, VKDF_LIGHT_DIRTY_SHADOWS);
}

inline void
vkdf_light_set_intensity(VkdfLight *l, float intensity)
{
   l->intensity = intensity;

   uint32_t dirty_bits = VKDF_LIGHT_DIRTY;
   bitfield_set(&l->dirty, dirty_bits);
}

inline float
vkdf_light_get_intensity(VkdfLight *l)
{
   return l->intensity;
}

inline void
vkdf_light_set_volume_scale_cap(VkdfLight *l, float cap)
{
   l->volume_scale_cap = cap;
}

glm::vec3
vkdf_light_get_volume_scale(VkdfLight *l);

inline void
vkdf_light_set_volume_cutoff(VkdfLight *l, float cutoff)
{
   assert(cutoff > 0.0f && cutoff <= 1.0f);
   l->volume_cutoff = cutoff;
}

inline float
vkdf_light_get_volume_cutoff(VkdfLight *l)
{
   return l->volume_cutoff;
}

void
vkdf_light_free(VkdfLight *light);

#endif
