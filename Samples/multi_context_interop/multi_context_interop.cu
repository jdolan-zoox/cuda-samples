#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#define CHECK(x) if(!(x)) printf("%s:%d %s\n", __FUNCTION__, __LINE__, #x)
#define CHECK_EQ(l,r) CHECK(l==r)

// EGL INCLUDES
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
// EGL includes some x11 headers that `#define Status int`, so we
// undefine that right after inclusion. This package is built with
// a define that turns x11 headers off, but YCM still complains about it
// unless we undefine it statically here.
#undef Status

#ifndef EGL_VERSION_1_5
#define EGL_VERSION_1_5 1
typedef intptr_t EGLAttrib;
#endif

#ifndef EGL_EXT_device_base
#define EGL_EXT_device_base 1
typedef void* EGLDeviceEXT;
#define EGL_NO_DEVICE_EXT EGL_CAST(EGLDeviceEXT, 0)
#define EGL_BAD_DEVICE_EXT 0x322B
#define EGL_DEVICE_EXT 0x322C
typedef EGLBoolean(EGLAPIENTRYP PFNEGLQUERYDEVICEATTRIBEXTPROC)(
    EGLDeviceEXT device, EGLint attribute, EGLAttrib* value);
typedef const char*(EGLAPIENTRYP PFNEGLQUERYDEVICESTRINGEXTPROC)(
    EGLDeviceEXT device, EGLint name);
typedef EGLBoolean(EGLAPIENTRYP PFNEGLQUERYDEVICESEXTPROC)(
    EGLint max_devices, EGLDeviceEXT* devices, EGLint* num_devices);
typedef EGLBoolean(EGLAPIENTRYP PFNEGLQUERYDISPLAYATTRIBEXTPROC)(
    EGLDisplay dpy, EGLint attribute, EGLAttrib* value);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglQueryDeviceAttribEXT(EGLDeviceEXT device,
                                                      EGLint attribute,
                                                      EGLAttrib* value);
EGLAPI const char* EGLAPIENTRY
    eglQueryDeviceStringEXT(EGLDeviceEXT device, EGLint name);
EGLAPI EGLBoolean EGLAPIENTRY eglQueryDevicesEXT(EGLint max_devices,
                                                 EGLDeviceEXT* devices,
                                                 EGLint* num_devices);
EGLAPI EGLBoolean EGLAPIENTRY eglQueryDisplayAttribEXT(EGLDisplay dpy,
                                                       EGLint attribute,
                                                       EGLAttrib* value);
#endif
#endif /* EGL_EXT_device_base */

#ifndef EGL_EXT_platform_base
#define EGL_EXT_platform_base 1
typedef EGLDisplay(EGLAPIENTRYP PFNEGLGETPLATFORMDISPLAYEXTPROC)(
    EGLenum platform, void* native_display, const EGLint* attrib_list);
typedef EGLSurface(EGLAPIENTRYP PFNEGLCREATEPLATFORMWINDOWSURFACEEXTPROC)(
    EGLDisplay dpy,
    EGLConfig config,
    void* native_window,
    const EGLint* attrib_list);
typedef EGLSurface(EGLAPIENTRYP PFNEGLCREATEPLATFORMPIXMAPSURFACEEXTPROC)(
    EGLDisplay dpy,
    EGLConfig config,
    void* native_pixmap,
    const EGLint* attrib_list);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLDisplay EGLAPIENTRY
    eglGetPlatformDisplayEXT(EGLenum platform,
                             void* native_display,
                             const EGLint* attrib_list);
EGLAPI EGLSurface EGLAPIENTRY
    eglCreatePlatformWindowSurfaceEXT(EGLDisplay dpy,
                                      EGLConfig config,
                                      void* native_window,
                                      const EGLint* attrib_list);
EGLAPI EGLSurface EGLAPIENTRY
    eglCreatePlatformPixmapSurfaceEXT(EGLDisplay dpy,
                                      EGLConfig config,
                                      void* native_pixmap,
                                      const EGLint* attrib_list);
#endif
#endif /* EGL_EXT_platform_base */

#ifndef EGL_EXT_platform_device
#define EGL_EXT_platform_device 1
#define EGL_PLATFORM_DEVICE_EXT 0x313F
#endif /* EGL_EXT_platform_device */
// END EGL INCLUDES

#include <GL/glew.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <memory>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <dlfcn.h>

static constexpr float kTextureFillValue[] = {0};

struct TopdownPainterVertex {
  float value;
  float x;
  float y;
  float z; // Must be <=0 to be visible in the scene.
  int channel;
};

const char* vertexShaderSource = R"(
#version 440
in vec3 vertex_position;
in float vertex_color;
in int channel_index;
uniform vec2 center;
uniform vec2 extent;
uniform float xy_rotation_degrees;
out float color;
out int layer;
void main() {
  color = vertex_color;
  layer = channel_index;
  float xy_rotation_radians = radians(xy_rotation_degrees);
  float cos_yaw = cos(xy_rotation_radians);
  float sin_yaw = sin(xy_rotation_radians);
  mat4 rotation_matrix = mat4(
    vec4(cos_yaw, -sin_yaw, 0, 0),
    vec4(sin_yaw, cos_yaw , 0, 0),
    vec4(0      , 0       , 1, 0),
    vec4(0      , 0       , 0, 1)
  );
  gl_Position = rotation_matrix*vec4(2*(vertex_position.x-center.x)/extent.x,
                                     2*(vertex_position.y-center.y)/extent.y,
                                     vertex_position.z,
                                     1.0);
}
)";

static const char* kTrianglesGeometryShaderText = R"(
#version 440
#extension GL_EXT_geometry_shader4 : enable
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
in int layer[];
in float color[];
out float vertex_color;
void main() {
    int i;
    for (i = 0; i < gl_VerticesIn; i++) {
      gl_Layer = layer[i];
      vertex_color = color[i];
      gl_Position = gl_PositionIn[i];
      EmitVertex();
    }
    EndPrimitive();
}
)";

const char* fragmentShaderSource =
    "#version 440\n"
    "in float vertex_color;\n"
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = vec4(vertex_color, 0.0, 0.0, 1.0f);\n"
    "}\n\0";

struct EglLib {
  void* lib = nullptr;

  decltype(::eglGetProcAddress)* getProcAddress = nullptr;
  decltype(::eglGetCurrentContext)* getCurrentContext = nullptr;
  decltype(::eglGetCurrentDisplay)* getCurrentDisplay = nullptr;
  decltype(::eglGetCurrentSurface)* getCurrentSurface = nullptr;
  decltype(::eglGetDisplay)* getDisplay = nullptr;
  decltype(::eglInitialize)* initialize = nullptr;
  decltype(::eglTerminate)* terminate = nullptr;
  decltype(::eglChooseConfig)* chooseConfig = nullptr;
  decltype(::eglCreatePbufferSurface)* createPbufferSurface = nullptr;
  decltype(::eglDestroySurface)* destroySurface = nullptr;
  decltype(::eglBindAPI)* bindAPI = nullptr;
  decltype(::eglCreateContext)* createContext = nullptr;
  decltype(::eglDestroyContext)* destroyContext = nullptr;
  decltype(::eglMakeCurrent)* makeCurrent = nullptr;
  decltype(::eglGetError)* getError = nullptr;

  PFNEGLQUERYDEVICESEXTPROC queryDevicesEXT = nullptr;
  PFNEGLGETPLATFORMDISPLAYEXTPROC getPlatformDisplayEXT = nullptr;
};

struct EglCudaInternalContext {
  EglLib egl;
  EGLDisplay egl_display;
  EGLSurface egl_surface;
  EGLContext egl_context;
};

void doInitializeGlew() {
  // Start GLEW extension handler
  glewExperimental = GL_TRUE;
  auto stat = glewInit();
  if (stat != GLEW_OK) {
    std::cout << "ERROR initializing GLEW: " << stat << std::endl;
    return;
  }

  // Get various version strings to print.
  const GLubyte* renderer = glGetString(GL_RENDERER);
  const GLubyte* version = glGetString(GL_VERSION);
  const GLubyte* glew_version = glewGetString(GLEW_VERSION);
  const GLubyte* glsl_version = glGetString(GL_SHADING_LANGUAGE_VERSION);
  if (glGetError() != GL_NO_ERROR) {
    std::cout << "ERROR initializing GLEW 2" << std::endl;
  }
}

template <typename T>
bool loadSymbol(EglLib& egl, T& sym, const char* name) {
  // Use getProcAddress if it's available.
  if (egl.getProcAddress != nullptr) {
    sym = reinterpret_cast<T>(egl.getProcAddress(name));
    if (sym != nullptr) {
      return true;
    }
  }

  // Otherwise, try to load directly.
  sym = reinterpret_cast<T>(dlsym(egl.lib, name));
  if (sym != nullptr) {
    return true;
  }

  std::cerr << "failed to load symbol: " << name << std::endl;
  return false;
}

bool loadSymbols(EglLib& egl) {
  bool ok = true;
  ok = ok && loadSymbol(egl, egl.getProcAddress, "eglGetProcAddress");
  ok = ok && loadSymbol(egl, egl.getCurrentContext, "eglGetCurrentContext");
  ok = ok && loadSymbol(egl, egl.getCurrentDisplay, "eglGetCurrentDisplay");
  ok = ok && loadSymbol(egl, egl.getCurrentSurface, "eglGetCurrentSurface");
  ok = ok && loadSymbol(egl, egl.getDisplay, "eglGetDisplay");
  ok = ok && loadSymbol(egl, egl.initialize, "eglInitialize");
  ok = ok && loadSymbol(egl, egl.terminate, "eglTerminate");
  ok = ok && loadSymbol(egl, egl.chooseConfig, "eglChooseConfig");
  ok = ok &&
      loadSymbol(egl, egl.createPbufferSurface, "eglCreatePbufferSurface");
  ok = ok && loadSymbol(egl, egl.destroySurface, "eglDestroySurface");
  ok = ok && loadSymbol(egl, egl.bindAPI, "eglBindAPI");
  ok = ok && loadSymbol(egl, egl.createContext, "eglCreateContext");
  ok = ok && loadSymbol(egl, egl.destroyContext, "eglDestroyContext");
  ok = ok && loadSymbol(egl, egl.makeCurrent, "eglMakeCurrent");
  ok = ok && loadSymbol(egl, egl.getError, "eglGetError");
  ok = ok && loadSymbol(egl, egl.queryDevicesEXT, "eglQueryDevicesEXT");
  ok = ok &&
      loadSymbol(egl, egl.getPlatformDisplayEXT, "eglGetPlatformDisplayEXT");
  return ok;
}

bool loadEgl(EglLib& egl) {
  egl.lib = dlopen("libEGL.so.1", RTLD_NOW);
  return loadSymbols(egl);
}

void initializeEgl(int device_idx,
                   int height,
                   int width,
                   EglCudaInternalContext& context) {
  static const EGLint config_attribs[] = {EGL_SURFACE_TYPE,
                                          EGL_PBUFFER_BIT,
                                          EGL_BLUE_SIZE,
                                          8,
                                          EGL_GREEN_SIZE,
                                          8,
                                          EGL_RED_SIZE,
                                          8,
                                          EGL_DEPTH_SIZE,
                                          8,
                                          EGL_RENDERABLE_TYPE,
                                          EGL_OPENGL_BIT,
                                          EGL_NONE

  };
  // 1. Select appropriate device.
  static const int kMaxDevices = 32;
  EGLDeviceEXT egl_devs[kMaxDevices];
  EGLint num_devices;

  if (!context.egl.queryDevicesEXT(kMaxDevices, egl_devs, &num_devices)) {
    std::cerr << "Couldn't query devices";
    return;
  }
  if (device_idx >= num_devices) {
    std::cerr << "Global device index out of bounds.";
    return;
  }
  context.egl_display = context.egl.getPlatformDisplayEXT(
      EGL_PLATFORM_DEVICE_EXT, egl_devs[device_idx], nullptr);
  if (context.egl_display == EGL_NO_DISPLAY) {
    std::cerr << "No display";
    return;
  }

  EGLint egl_major, egl_minor, egl_error;
  if (!context.egl.initialize(context.egl_display, &egl_major, &egl_minor)) {
    egl_error = context.egl.getError();
    if (egl_error != EGL_SUCCESS) {
      std::cerr << "Failed to initialize EGL (" << egl_error << ").";
    }
    return;
  }

  // 2. Select an appropriate configuration
  EGLConfig egl_config;
  EGLint num_configs;
  if (!context.egl.chooseConfig(
          context.egl_display, config_attribs, &egl_config, 1, &num_configs)) {
    egl_error = context.egl.getError();
    if (egl_error != EGL_SUCCESS) {
      std::cerr << "Failed to choose config (" << egl_error << ").";
    }
    return;
  }

  // 3. Create a surface
  const EGLint pbuffer_attribs[] = {
      EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE,
  };

  context.egl_surface = context.egl.createPbufferSurface(
      context.egl_display, egl_config, pbuffer_attribs);
  if (context.egl_surface == EGL_NO_SURFACE) {
    egl_error = eglGetError();
    if (egl_error != EGL_SUCCESS) {
      std::cerr << "Failed to create surface (" << egl_error << ").";
    }
    return;
  }

  // 4. Bind the API.
  if (!context.egl.bindAPI(EGL_OPENGL_API)) {
    egl_error = context.egl.getError();
    if (egl_error != EGL_SUCCESS) {
      std::cerr << "Failed to bind API (" << egl_error << ").";
    }
    return;
  }

  // 5. Create a context.
  context.egl_context = context.egl.createContext(
      context.egl_display, egl_config, EGL_NO_CONTEXT, nullptr);
  if (context.egl_context == EGL_NO_CONTEXT) {
    egl_error = context.egl.getError();
    if (egl_error != EGL_SUCCESS) {
      std::cerr << "Failed create context (" << egl_error << ").";
    }
    return;
  }
}

void cudaStatus(cudaError err, const std::string& failed_call) {
  if (err != cudaSuccess) {
    std::cout << "Function " << failed_call
              << " failed: " << cudaGetErrorName(err) << " ("
              << cudaGetErrorString(err) << ")" << std::endl;
  }
}

std::mutex gui_lock;
std::condition_variable gui_ready_trigger;
bool gui_ready = false;
std::atomic<bool> should_quit(false);

void signalGuiReady() {
  std::lock_guard<std::mutex> lk(gui_lock);
  gui_ready = true;
  gui_ready_trigger.notify_one();
}

void waitForGuiReady() {
  std::unique_lock<std::mutex> lk(gui_lock);
  gui_ready_trigger.wait(lk, []{return gui_ready;});
}

void runFakeGui() {
  static constexpr EGLint kEglConfigAttrs[] = {EGL_SURFACE_TYPE,
                                             EGL_PBUFFER_BIT,
                                             EGL_BLUE_SIZE,
                                             8,
                                             EGL_GREEN_SIZE,
                                             8,
                                             EGL_RED_SIZE,
                                             8,
                                             EGL_DEPTH_SIZE,
                                             8,
                                             EGL_RENDERABLE_TYPE,
                                             EGL_OPENGL_BIT,
                                             EGL_NONE};
  static const EGLint kEglPBufferAttrs[] = {
      EGL_WIDTH, 64, EGL_HEIGHT, 64, EGL_NONE,
  };
  EglLib egl;
  loadEgl(egl);
  EGLDisplay display = nullptr;
  if(std::getenv("DISPLAY")) {
    // These should be roughly equivalent, and both fail in the same way
    display = egl.getDisplay(EGL_DEFAULT_DISPLAY);
    //#define EGL_PLATFORM_X11_EXT 0x31D5
    //display = egl.getPlatformDisplayEXT(EGL_PLATFORM_X11_EXT, XOpenDisplay(nullptr), nullptr);
  } else {
    static const int kMaxDevices = 32;
    EGLDeviceEXT egl_devs[kMaxDevices];
    EGLint num_devices = 0;
    CHECK(egl.queryDevicesEXT(kMaxDevices, egl_devs, &num_devices));
    display = egl.getPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, egl_devs[0], nullptr);
  }
  CHECK(egl.initialize(display, nullptr, nullptr) == EGL_TRUE);
  CHECK(egl.bindAPI(EGL_OPENGL_API) == EGL_TRUE);
  EGLConfig config = nullptr;
  int config_count = 0;
  CHECK(egl.chooseConfig(display,
                        kEglConfigAttrs,
                        &config,
                        1,
                        &config_count) == EGL_TRUE);
  CHECK(config_count == 1);
  auto context = egl.createContext(display, config, EGL_NO_CONTEXT, nullptr);
  auto surface = egl.createPbufferSurface(display, config, kEglPBufferAttrs);
  CHECK(surface != EGL_NO_SURFACE);
  CHECK(egl.makeCurrent(display, surface, surface, context));
  auto* gl_genTextures = reinterpret_cast<decltype(&glGenTextures)>(egl.getProcAddress("glGenTextures"));
  auto* gl_bindTexture = reinterpret_cast<decltype(&glBindTexture)>(egl.getProcAddress("glBindTexture"));
  auto* gl_texImage2D = reinterpret_cast<decltype(&glTexImage2D)>(egl.getProcAddress("glTexImage2D"));
  auto* gl_finish = reinterpret_cast<decltype(&glFinish)>(egl.getProcAddress("glFinish"));
  auto* gl_deleteTextures = reinterpret_cast<decltype(&glDeleteTextures)>(egl.getProcAddress("glDeleteTextures"));
  signalGuiReady();
  while(should_quit.load() == false) {
    GLuint texture = 0;
    gl_genTextures(1, &texture);
    gl_bindTexture(GL_TEXTURE_2D, texture);
    std::vector<uint8_t> img(1024*1024*4);
    gl_texImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 1024,
                 1024,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 img.data());
    gl_finish();
    gl_deleteTextures(1, &texture);
  }
}

int main(int argc, char* argv[]) {
  auto gui_thread = std::thread(runFakeGui);
  waitForGuiReady();

  EglCudaInternalContext context;
  if (!loadEgl(context.egl)) {
    std::cerr << "Failed to load EGL." << std::endl;
    return -1;
  }

  // cudaSetDevice(0);
  int height = 449;
  int width = 449;
  int depth = 3;

  static constexpr int kVertexPositionUtmIdx = 0;
  static constexpr int kColorIdx = 1;
  static constexpr int kChannelIndexIdx = 2;

  for (int i = 0; i < 1; i++) {
    initializeEgl(0, height, width, context);
    if (!context.egl.makeCurrent(context.egl_display,
                                 context.egl_surface,
                                 context.egl_surface,
                                 context.egl_context)) {
      std::cerr << "failed to attach context" << std::endl;
    }

    doInitializeGlew();
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_3D, texture);

    // 3. Tell GL how to filter geometry in that texture.
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    CHECK_EQ(glGetError(), GL_NO_ERROR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    glTexImage3D(GL_TEXTURE_3D,
                 0,
                 GL_R32F,
                 width,
                 height,
                 depth,
                 0,
                 GL_RED,
                 GL_FLOAT,
                 nullptr);

    // TO REPRODUCE ISSUE COMMENT OUT 7 lines below!
    // glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0);
    // CHECK_EQ(glGetError(), GL_NO_ERROR);

    // glClearTexImage(texture, 0, GL_RED, GL_FLOAT, kTextureFillValue);
    // CHECK_EQ(glGetError(), GL_NO_ERROR);
    // END REPRODUCE ISSUE BLOCK

    GLuint depth_texture;
    glGenTextures(1, &depth_texture);
    glBindTexture(GL_TEXTURE_2D_ARRAY, depth_texture);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    // 6. Set some filtering and wrapping parameters on the texture.
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    CHECK_EQ(glGetError(), GL_NO_ERROR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    CHECK_EQ(glGetError(), GL_NO_ERROR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    CHECK_EQ(glGetError(), GL_NO_ERROR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    CHECK_EQ(glGetError(), GL_NO_ERROR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    // 7. Back the depth texture with a multi-channel image.
    glTexImage3D(GL_TEXTURE_2D_ARRAY,
                 0,
                 GL_DEPTH_COMPONENT32,
                 width,
                 height,
                 depth,
                 0,
                 GL_DEPTH_COMPONENT,
                 GL_FLOAT,
                 nullptr);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);

    // 8. Associate the depth texture to the framebuffer as its depth
    // attachment.
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth_texture, 0);

    cudaStream_t render_stream;
    cudaStreamCreateWithFlags(&render_stream, cudaStreamNonBlocking);
    cudaGraphicsResource_t cuda_texture;
    cudaStatus(cudaGraphicsGLRegisterImage(&cuda_texture,
                                           texture,
                                           GL_TEXTURE_3D,
                                           cudaGraphicsRegisterFlagsNone),
               "cudaGraphicsGLRegisterImage");
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        80,
        20,
        0.0f, // top right
        80,
        80,
        0.0f, // bottom right
        20,
        80,
        0.0f, // bottom left
        20,
        20,
        0.0f // top left
    };
    unsigned int indices[] = {
        // note that we start from 0!
        3,
        2,
        0, // first Triangle
        2,
        1,
        0 // second Triangle
    };
    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s),
    // and
    // then configure vertex attributes(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    static constexpr unsigned kDataBufferIdx = 0;
    glBindVertexBuffer(kDataBufferIdx,
                       VBO,
                       0,
                       // TODO: if we factor out the draw call (render to all
                       // layers of the 3D texture simultaneously) we should
                       // use `decltype(vertices)::value_type` here.
                       sizeof(TopdownPainterVertex));

    GLbitfield buffer_flags =
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    size_t buffer_size = 512 * sizeof(float);
    glBufferStorage(GL_ARRAY_BUFFER, buffer_size, nullptr, buffer_flags);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    // 15. Map the vertex buffer object (where painters will be writing their
    // vertices) to the CPU.
    auto mapped_vbo = reinterpret_cast<TopdownPainterVertex*>(
        glMapBufferRange(GL_ARRAY_BUFFER, 0, buffer_size, buffer_flags));
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    glVertexAttribFormat(kVertexPositionUtmIdx,
                         3,
                         GL_FLOAT,
                         GL_FALSE,
                         offsetof(TopdownPainterVertex, x));
    glEnableVertexAttribArray(kVertexPositionUtmIdx);
    glVertexAttribBinding(kVertexPositionUtmIdx, kDataBufferIdx);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    // Attribute 1 is color.
    glVertexAttribFormat(kColorIdx,
                         1,
                         GL_FLOAT,
                         GL_FALSE,
                         offsetof(TopdownPainterVertex, value));
    glEnableVertexAttribArray(kColorIdx);
    glVertexAttribBinding(kColorIdx, kDataBufferIdx);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    // Attribute 2 is layer.
    glVertexAttribIFormat(
        kChannelIndexIdx, 1, GL_INT, offsetof(TopdownPainterVertex, channel));
    glEnableVertexAttribArray(kChannelIndexIdx);
    glVertexAttribBinding(kChannelIndexIdx, kDataBufferIdx);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    // build and compile our shader program
    // ------------------------------------
    // vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
      std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog
                << std::endl;
    }

    // Geometry shader
    GLuint geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(geometry_shader, 1, &kTrianglesGeometryShaderText, NULL);
    glCompileShader(geometry_shader);
    int compiled_successfully;
    glGetShaderiv(geometry_shader, GL_COMPILE_STATUS, &compiled_successfully);
    if (compiled_successfully == GL_FALSE) {
      GLint max_length = 0;
      glGetShaderiv(geometry_shader, GL_INFO_LOG_LENGTH, &max_length);
      // The max_length includes the NULL character
      GLchar errorLog[max_length];
      glGetShaderInfoLog(
          geometry_shader, max_length, &max_length, &errorLog[0]);
      printf("Geometry shader compilation failed: %s\n", errorLog);
    }

    // fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
      std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog
                << std::endl;
    }

    // link shaders
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    CHECK_EQ(glGetError(), GL_NO_ERROR);
    glAttachShader(shaderProgram, fragmentShader);
    CHECK_EQ(glGetError(), GL_NO_ERROR);
    glAttachShader(shaderProgram, geometry_shader);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    glBindAttribLocation(
        shaderProgram, kVertexPositionUtmIdx, "vertex_position");
    CHECK_EQ(glGetError(), GL_NO_ERROR);
    glBindAttribLocation(shaderProgram, kColorIdx, "vertex_color");
    CHECK_EQ(glGetError(), GL_NO_ERROR);
    glBindAttribLocation(shaderProgram, kChannelIndexIdx, "channel_index");
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    glLinkProgram(shaderProgram);

    // check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
      glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
      std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog
                << std::endl;
    }

    CHECK_EQ(glGetAttribLocation(shaderProgram, "channel_index"),
             kChannelIndexIdx);
    CHECK_EQ(glGetAttribLocation(shaderProgram, "vertex_color"),
             kColorIdx);
    CHECK_EQ(glGetAttribLocation(shaderProgram, "vertex_position"),
             kVertexPositionUtmIdx);

    auto center_location = glGetUniformLocation(shaderProgram, "center");
    CHECK_EQ(glGetError(), GL_NO_ERROR);
    auto extent_location = glGetUniformLocation(shaderProgram, "extent");
    CHECK_EQ(glGetError(), GL_NO_ERROR);
    auto xy_rotation_degrees_location =
        glGetUniformLocation(shaderProgram, "xy_rotation_degrees");


    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    CHECK_EQ(glGetError(), GL_NO_ERROR);
    glBindTexture(GL_TEXTURE_3D, 0);
    CHECK_EQ(glGetError(), GL_NO_ERROR);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    // Detach
    CHECK(context.egl.makeCurrent(
        context.egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT));

    // Re attach
    if (!context.egl.makeCurrent(context.egl_display,
                                 context.egl_surface,
                                 context.egl_surface,
                                 context.egl_context)) {
      std::cerr << "failed to attach context" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    // Load attributes and other rendering configuration.
    glBindVertexArray(VAO);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    glClearTexImage(texture, 0, GL_RED, GL_FLOAT, kTextureFillValue);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    // render
    // ------
    float color[] = {0.8, 0.2, 0.5};

    for (int c = 0; c < 3; c++) {
      for (int i = 0; i < 6; i++) {
        mapped_vbo[i + c*6].value = color[c];
        mapped_vbo[i + c*6].x = vertices[indices[i] * 3];
        mapped_vbo[i + c*6].y = vertices[indices[i] * 3 + 1];
        mapped_vbo[i + c*6].z = vertices[indices[i] * 3 + 2];
        mapped_vbo[i + c*6].channel = c;
      }
    }

    // draw our first triangle
    glUseProgram(shaderProgram);

    glUniform2f(center_location, 50, 50);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    glUniform2f(extent_location,
                100,
                100);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    glUniform1f(xy_rotation_degrees_location, 0.0);
    CHECK_EQ(glGetError(), GL_NO_ERROR);

    glBindVertexArray(
        VAO); // seeing as we only have a single VAO there's no need
              // to bind it every time, but we'll do so to keep
              // things a bit more organized
    glClearDepth(-1.0);
    glClear(GL_DEPTH_BUFFER_BIT);

    // glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glDrawArrays(GL_TRIANGLES, 0, 18);
    CHECK_EQ(glGetError(), GL_NO_ERROR);
    // glBindVertexArray(0); // no need to unbind it every time

    glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glUseProgram(0);

    cudaArray_t array;
    cudaChannelFormatDesc desc;
    cudaExtent extent;

    cudaStatus(cudaGraphicsMapResources(1, &cuda_texture, render_stream),
               "cudaGraphicsMapResources");
    cudaStatus(
        cudaGraphicsSubResourceGetMappedArray(&array, cuda_texture, 0, 0),
        "cudaGraphicsSubResourceGetMappedArray");
    cudaStatus(cudaArrayGetInfo(&desc, &extent, NULL, array),
               "cudaArrayGetInfo");

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.addressMode[2] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    std::vector<float> data(extent.height * extent.width * extent.depth);
    std::vector<float> data_gl(extent.height * extent.width * extent.depth);

    // Copy from device
    cudaMemcpy3DParms copyParams = {0};
    copyParams.dstPtr = make_cudaPitchedPtr(
        data.data(), extent.width * sizeof(float), extent.width, extent.height);
    copyParams.srcArray = array;
    copyParams.kind = cudaMemcpyDeviceToHost;
    copyParams.extent = extent;
    cudaStatus(cudaMemcpy3DAsync(&copyParams, render_stream), "memcpy3d");
    cudaStreamSynchronize(render_stream);

    glBindTexture(GL_TEXTURE_3D, texture);
    glGetTexImage(GL_TEXTURE_3D,
                   0,
                   GL_RED,
                   GL_FLOAT,
                   data_gl.data());
    bool all_equal = true;
    for (size_t i = 0; i < data.size(); i++) {
      if (data[i] != data_gl[i]) {
        all_equal = false;
        break;
      }
    }

    cudaStatus(cudaGraphicsUnmapResources(1, &cuda_texture),
               "cudaGraphicsUnmapResources");
    context.egl.terminate(context.egl_display);
    context.egl.destroySurface(context.egl_display, context.egl_surface);
    context.egl.destroyContext(context.egl_display, context.egl_context);
    dlclose(context.egl.lib);
    if (!all_equal)
      std::cout << height << " " << (all_equal ? "Pass" : "Fail") << std::endl;

    height++;
    width++;
  }

  should_quit.store(true);
  gui_thread.join();

  return 0;
}
