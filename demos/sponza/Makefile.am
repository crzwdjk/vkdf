bin_PROGRAMS = sponza

AM_CPPFLAGS = @DEMO_DEPS_CFLAGS@

# ------------------------------
# Sponza
# ------------------------------

BUILT_SOURCES = \
    obj.depthprepass.vert.spv \
    obj.depthprepass.opacity.vert.spv \
    obj.depthprepass.opacity.frag.spv \
    obj.vert \
    obj.vert.spv \
    obj.frag \
    obj.frag.spv \
    obj_opacity.frag \
    obj_opacity.frag.spv \
    obj.deferred.vert \
    obj.deferred.vert.spv \
    obj.deferred.frag \
    obj.deferred.frag.spv \
    obj.deferred-float16.frag \
    obj.deferred-float16.frag.spv \
    obj_opacity.deferred.frag \
    obj_opacity.deferred.frag.spv \
    obj_opacity.deferred-float16.frag \
    obj_opacity.deferred-float16.frag.spv \
    gbuffer-merge.vert.spv \
    gbuffer-merge.frag \
    gbuffer-merge.frag.spv \
    gbuffer-merge-float16.frag \
    gbuffer-merge-float16.frag.spv \
    gbuffer-merge.ssao.frag \
    gbuffer-merge.ssao.frag.spv \
    gbuffer-merge-float16.ssao.frag \
    gbuffer-merge-float16.ssao.frag.spv \
    debug-tile.vert.spv \
    debug-tile.frag.spv

CLEANFILES = \
    $(BUILT_SOURCES)

LIGHTING_GLSL = ../../data/glsl/lighting.glsl
UTIL_GLSL = ../../data/glsl/util.glsl
LIGHTING_FLOAT16_GLSL = ../../data/glsl/lighting-float16.glsl

%.frag.spv: %.frag
	$(top_srcdir)/$(GLSLANG) -V $< -o $@

%.vert.spv: %.vert
	$(top_srcdir)/$(GLSLANG) -V $< -o $@

%.vert: %.vert.input
	python $(top_srcdir)/scripts/fixup-glsl.py $< $@

%.frag: %.frag.input
	python $(top_srcdir)/scripts/fixup-glsl.py $< $@

# Forward shading
obj.vert: obj.vert.input $(LIGTHING_GLSL)

obj.frag: obj.frag.input $(LIGHTING_GLSL) compute_forward_lighting.glsl

obj_opacity.frag: obj_opacity.frag.input

# Deferred shading
obj.deferred.vert: obj.deferred.vert.input $(LIGHTING_GLSL)

obj.deferred.frag: obj.deferred.frag.input $(LIGHTING_GLSL) compute_gbuffer.glsl

obj.deferred-float16.frag: obj.deferred-float16.frag.input \
    $(LIGHTING_FLOAT16_GLSL) compute_gbuffer_float16.glsl

obj_opacity.deferred.frag: obj_opacity.deferred.frag.input $(LIGHTING_GLSL) \
    compute_gbuffer.glsl

obj_opacity.deferred.frag: obj_opacity.deferred.frag.input \
    $(LIGHTING_FLOAT16_GLSL) compute_gbuffer_float16.glsl

gbuffer-merge.frag: gbuffer-merge.frag.input $(LIGHTING_GLSL) \
    $(UTIL_GLSL) compute_deferred_lighting.glsl

gbuffer-merge-float16.frag: gbuffer-merge-float16.frag.input \
    $(LIGHTING_FLOAT16_GLSL) compute_deferred_lighting.glsl

# SSAO (deferred)
gbuffer-merge.ssao.frag: gbuffer-merge.ssao.frag.input $(LIGHTING_GLSL) \
    $(UTIL_GLSL) compute_deferred_lighting.glsl

gbuffer-merge-float16.ssao.frag: gbuffer-merge-float16.ssao.frag.input \
    $(LIGHTING_GLSL) $(UTIL_GLSL) compute_deferred_lighting_float16.glsl

sponza_SOURCES = \
    main.cpp

sponza_CXXFLAGS = \
    -DPREFIX=$(prefix) \
    -D_GNU_SOURCE \
    @VKDF_DEFINES@

sponza_LDADD = \
    $(abs_top_builddir)/framework/.libs/libvkdf.so \
    @DEMO_DEPS_LIBS@ \
    -lm

# -----------------------------

MAINTAINERCLEANFILES = \
	*.in \
	*~

DISTCLEANFILES = $(MAINTAINERCLEANFILES)
