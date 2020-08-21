//---------------------------------//
//  This file is part of MuJoCo    //
//  Written by Emo Todorov         //
//  Copyright (C) 2017 Roboti LLC  //
//---------------------------------//


#pragma once


#define mjNGROUP        5           // number of geom and site groups with visflags
#define mjMAXOVERLAY    500         // maximum number of characters in overlay text
#define mjMAXLINE       100         // maximum number of lines per plot
#define mjMAXLINEPNT    500         // maximum number points per line
#define mjMAXPLANEGRID  100         // maximum number of grid points for plane rendering


typedef enum _mjtCatBit             // bitflags for mjvGeom category
{
    mjCAT_STATIC        = 1,        // model elements in body 0
    mjCAT_DYNAMIC       = 2,        // model elements in all other bodies
    mjCAT_DECOR         = 4,        // decorative geoms
    mjCAT_ALL           = 7         // select all categories
} mjtCatBit;


typedef enum _mjtMouse              // mouse interaction mode
{
    mjMOUSE_NONE         = 0,       // no action
    mjMOUSE_ROTATE_V,               // rotate, vertical plane
    mjMOUSE_ROTATE_H,               // rotate, horizontal plane
    mjMOUSE_MOVE_V,                 // move, vertical plane
    mjMOUSE_MOVE_H,                 // move, horizontal plane
    mjMOUSE_ZOOM,                   // zoom
    mjMOUSE_SELECT                  // selection
} mjtMouse;


typedef enum _mjtPertBit            // mouse perturbations
{
    mjPERT_TRANSLATE    = 1,        // translation
    mjPERT_ROTATE       = 2         // rotation
} mjtPertBit;


typedef enum _mjtCamera             // abstract camera type
{
    mjCAMERA_FREE        = 0,       // free camera
    mjCAMERA_TRACKING,              // tracking camera; uses trackbodyid 
    mjCAMERA_FIXED,                 // fixed camera; uses fixedcamid
    mjCAMERA_USER                   // user is responsible for setting OpenGL camera
} mjtCamera;


typedef enum _mjtLabel              // object labeling
{
    mjLABEL_NONE        = 0,        // nothing
    mjLABEL_BODY,                   // body labels
    mjLABEL_JOINT,                  // joint labels
    mjLABEL_GEOM,                   // geom labels
    mjLABEL_SITE,                   // site labels
    mjLABEL_CAMERA,                 // camera labels
    mjLABEL_LIGHT,                  // light labels
    mjLABEL_TENDON,                 // tendon labels
    mjLABEL_ACTUATOR,               // actuator labels
    mjLABEL_CONSTRAINT,             // constraint labels
    mjLABEL_SELECTION,              // selected object
    mjLABEL_SELPNT,                 // coordinates of selection point
    mjLABEL_CONTACTFORCE,           // magnitude of contact force

    mjNLABEL                        // number of label types
} mjtLabel;


typedef enum _mjtFrame              // frame visualization
{
    mjFRAME_NONE        = 0,        // no frames
    mjFRAME_BODY,                   // body frames
    mjFRAME_GEOM,                   // geom frames
    mjFRAME_SITE,                   // site frames
    mjFRAME_CAMERA,                 // camera frames
    mjFRAME_LIGHT,                  // light frames
    mjFRAME_WORLD,                  // world frame

    mjNFRAME                        // number of visualization frames
} mjtFrame;


typedef enum _mjtVisFlag            // flags enabling model element visualization
{
    mjVIS_CONVEXHULL    = 0,        // mesh convex hull
    mjVIS_TEXTURE,                  // textures
    mjVIS_JOINT,                    // joints
    mjVIS_ACTUATOR,                 // actuators
    mjVIS_CAMERA,                   // cameras
    mjVIS_LIGHT,                    // lights
    mjVIS_CONSTRAINT,               // point constraints
    mjVIS_INERTIA,                  // equivalent inertia boxes
    mjVIS_PERTFORCE,                // perturbation force
    mjVIS_PERTOBJ,                  // perturbation object
    mjVIS_CONTACTPOINT,             // contact points
    mjVIS_CONTACTFORCE,             // contact force
    mjVIS_CONTACTSPLIT,             // split contact force into normal and tanget
    mjVIS_TRANSPARENT,              // make dynamic geoms more transparent
    mjVIS_AUTOCONNECT,              // auto connect joints and body coms
    mjVIS_COM,                      // center of mass
    mjVIS_SELECT,                   // selection point
    mjVIS_STATIC,                   // static bodies

    mjNVISFLAG                      // number of visualization flags
} mjtVisFlag;


typedef enum _mjtRndFlag            // flags enabling rendering effects
{
    mjRND_SHADOW        = 0,        // shadows
    mjRND_WIREFRAME,                // wireframe
    mjRND_REFLECTION,               // reflections
    mjRND_FOG,                      // fog
    mjRND_SKYBOX,                   // skybox

    mjNRNDFLAG                      // number of rendering flags
} mjtRndFlag;


typedef enum _mjtStereo             // type of stereo rendering
{
    mjSTEREO_NONE       = 0,        // no stereo; use left eye only
    mjSTEREO_QUADBUFFERED,          // quad buffered; revert to side-by-side if no hardware support
    mjSTEREO_SIDEBYSIDE             // side-by-side
} mjtStereo;


struct _mjvPerturb                  // object selection and perturbation
{
    int      select;                // selected body id; non-positive: none
    int      active;                // perturbation bitmask (mjtPertBit)
    mjtNum   refpos[3];             // desired position for selected object
    mjtNum   refquat[4];            // desired orientation for selected object
    mjtNum   localpos[3];           // selection point in object coordinates
    mjtNum   scale;                 // relative mouse motion-to-space scaling (set by initPerturb)
};
typedef struct _mjvPerturb mjvPerturb;


struct _mjvCamera                   // abstract camera
{
    // type and ids
    int      type;                  // camera type (mjtCamera)
    int      fixedcamid;            // fixed camera id
    int      trackbodyid;           // body id to track

    // abstract camera pose specification
    mjtNum   lookat[3];             // lookat point
    mjtNum   distance;              // distance to lookat point or tracked body
    mjtNum   azimuth;               // camera azimuth (deg)
    mjtNum   elevation;             // camera elevation (deg)
};
typedef struct _mjvCamera mjvCamera;


struct _mjvGLCamera                 // OpenGL camera
{
    // camera frame
    float    pos[3];                // position
    float    forward[3];            // forward direction
    float    up[3];                 // up direction

    // camera projection
    float    frustum_center;        // hor. center (left,right set to match aspect)
    float    frustum_bottom;        // bottom
    float    frustum_top;           // top
    float    frustum_near;          // near
    float    frustum_far;           // far
};
typedef struct _mjvGLCamera mjvGLCamera;


struct _mjvGeom                     // abstract geom
{
    // type info
    int      type;                  // geom type (mjtGeom)
    int      dataid;                // mesh, hfield or plane id; -1: none
    int      objtype;               // mujoco object type; mjOBJ_UNKNOWN for decor
    int      objid;                 // mujoco object id; -1 for decor
    int      category;              // visual category
    int      texid;                 // texture id; -1: no texture
    int      texuniform;            // uniform cube mapping

    // OpenGL info
    float    texrepeat[2];          // texture repetition for 2D mapping
    float    size[3];               // size parameters
    float    pos[3];                // Cartesian position
    float    mat[9];                // Cartesian orientation
    float    rgba[4];               // color and transparency
    float    emission;              // emission coef
    float    specular;              // specular coef
    float    shininess;             // shininess coef
    float    reflectance;           // reflectance coef
    char     label[100];            // text label

    // transparency rendering (set internally)
    float    camdist;               // distance to camera (used by sorter)
    float    modelrbound;           // geom rbound from model, 0 if not model geom
    mjtByte  transparent;           // treat geom as transparent
};
typedef struct _mjvGeom mjvGeom;


struct _mjvLight                    // OpenGL light
{
    float    pos[3];                // position rel. to body frame              
    float    dir[3];                // direction rel. to body frame             
    float    attenuation[3];        // OpenGL attenuation (quadratic model)     
    float    cutoff;                // OpenGL cutoff                            
    float    exponent;              // OpenGL exponent                          
    float    ambient[3];            // ambient rgb (alpha=1)                    
    float    diffuse[3];            // diffuse rgb (alpha=1)                    
    float    specular[3];           // specular rgb (alpha=1)
    mjtByte  headlight;             // headlight
    mjtByte  directional;           // directional light                        
    mjtByte  castshadow;            // does light cast shadows                  
};
typedef struct _mjvLight mjvLight;


struct _mjvOption                   // abstract visualization options
{
    int      label;                 // what objects to label (mjtLabel)
    int      frame;                 // which frame to show (mjtFrame)
    mjtByte  geomgroup[mjNGROUP];   // geom visualization by group
    mjtByte  sitegroup[mjNGROUP];   // site visualization by group
    mjtByte  flags[mjNVISFLAG];     // visualization flags (indexed by mjtVisFlag)
};
typedef struct _mjvOption mjvOption;


struct _mjvScene                    // abstract scene passed to OpenGL renderer
{
    // abstract geoms
    int      maxgeom;               // size of allocated geom buffer
    int      ngeom;                 // number of geoms currently in buffer
    mjvGeom* geoms;                 // buffer for geoms
    int*     geomorder;             // buffer for ordering geoms by distance to camera

    // OpenGL lights
    int      nlight;                // number of lights currently in buffer
    mjvLight lights[8];             // buffer for lights

    // OpenGL cameras
    mjvGLCamera camera[2];          // left and right camera

    // OpenGL model transformation
    mjtByte  enabletransform;       // enable model transformation
    float    translate[3];          // model translation
    float    rotate[4];             // model quaternion rotation
    float    scale;                 // model scaling

    // OpenGL rendering effects
    int      stereo;                // stereoscopic rendering (mjtStereo)
    mjtByte  flags[mjNRNDFLAG];     // rendering flags (indexed by mjtRndFlag)
};
typedef struct _mjvScene mjvScene;


struct _mjvFigure                   // abstract 2D figure passed to OpenGL renderer
{
    // enable/disable flags
    int     flg_legend;             // show legend
    int     flg_ticklabel[2];       // show grid tick labels (x,y)
    int     flg_extend;             // automatically extend axis ranges to fit data
    int     flg_barplot;            // isolated line segments (i.e. GL_LINES)

    // figure options
    int     gridsize[2];            // number of grid points in (x,y)
    float   gridrgb[3];             // grid line rgb
    float   gridwidth;              // grid line width
    float   figurergba[4];          // figure color and alpha
    float   legendrgba[4];          // legend color and alpha
    float   textrgb[3];             // text color
    float   range[2][2];            // axis ranges; (min>=max) automatic
    char    xlabel[100];            // x-axis label
    char    title[100];             // figure title
    char    xformat[20];            // x-tick label format for sprintf
    char    yformat[20];            // y-tick label format for sprintf
    char    minwidth[20];           // string used to determine min y-tick width

    // line data
    int     linepnt[mjMAXLINE];                   // number of points in line; (0) disable
    float   linergb[mjMAXLINE][3];                // line color
    float   linewidth[mjMAXLINE];                 // line width
    float   linedata[mjMAXLINE][2*mjMAXLINEPNT];  // line data (x,y)
    char    linename[mjMAXLINE][100];             // line name for legend
};
typedef struct _mjvFigure mjvFigure;
