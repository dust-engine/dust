#                     GLSL.std.450              	       main                                      draw.playout     >    �     #version 460
#include "draw.playout"

layout(location = 0) in vec4 in_Color;
layout(location = 1) in vec2 in_UV;
layout(location = 0) out vec4 out_Color;

void main(){
    out_Color = in_Color * texture(u_img, in_UV);
}
   $    �     layout (binding = 0u, set = 0u) uniform sampler2D u_img;
layout (push_constant) uniform PushConstants {vec2 u_transform;

};
    
 GL_GOOGLE_cpp_style_line_directive    GL_GOOGLE_include_directive      main         out_Color        in_Color         u_img        in_UV   J entry-point main    J client vulkan100    J target-env spirv1.5 J target-env vulkan1.2    J entry-point main    G            G            G     "       G     !       G                !                   	            
      	   ;  
                  	   ;            	                                                  ;                                   ;                      6               �          	       =  	         =           =           W  	            �  	            >        �  8  