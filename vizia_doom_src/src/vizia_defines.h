#ifndef __VIZIA_DEFINES_H__
#define __VIZIA_DEFINES_H__

#include <stdlib.h>

#define VIZIA_BT_ATTACK 0
#define VIZIA_BT_USE 1
#define VIZIA_BT_JUMP 2
#define VIZIA_BT_CROUCH 3
#define VIZIA_BT_TURN180 4
#define VIZIA_BT_ALTATTACK 5
#define VIZIA_BT_RELOAD 6
#define VIZIA_BT_ZOOM 7

#define VIZIA_BT_SPEED 8
#define VIZIA_BT_STRAFE 9

#define VIZIA_BT_MOVE_RIGHT 10
#define VIZIA_BT_MOVE_LEFT 11
#define VIZIA_BT_MOVE_BACK 12
#define VIZIA_BT_MOVE_FORWARD 13
#define VIZIA_BT_TURN_RIGHT 14
#define VIZIA_BT_TURN_LEFT 15
#define VIZIA_BT_LOOK_UP 16
#define VIZIA_BT_LOOK_DOWN 17
#define VIZIA_BT_MOVE_UP 18
#define VIZIA_BT_MOVE_DOWN 19
#define VIZIA_BT_LAND 20
//#define VIZIA_BT_SHOWSCORES 20

#define VIZIA_BT_SELECT_WEAPON1 21
#define VIZIA_BT_SELECT_WEAPON2 22
#define VIZIA_BT_SELECT_WEAPON3 23
#define VIZIA_BT_SELECT_WEAPON4 24
#define VIZIA_BT_SELECT_WEAPON5 25
#define VIZIA_BT_SELECT_WEAPON6 26
#define VIZIA_BT_SELECT_WEAPON7 27
#define VIZIA_BT_SELECT_WEAPON8 28
#define VIZIA_BT_SELECT_WEAPON9 29
#define VIZIA_BT_SELECT_WEAPON0 30

#define VIZIA_BT_SELECT_NEXT_WEAPON 31
#define VIZIA_BT_SELECT_PREV_WEAPON 32
#define VIZIA_BT_DROP_SELECTED_WEAPON 33

#define VIZIA_BT_ACTIVATE_SELECTED_ITEM 34
#define VIZIA_BT_SELECT_NEXT_ITEM 35
#define VIZIA_BT_SELECT_PREV_ITEM 36
#define VIZIA_BT_DROP_SELECTED_ITEM 37

#define VIZIA_BT_VIEW_PITCH 38
#define VIZIA_BT_VIEW_ANGLE 39
#define VIZIA_BT_FORWARD_BACKWARD 40
#define VIZIA_BT_LEFT_RIGHT 41
#define VIZIA_BT_UP_DOWN 42

#define VIZIA_BT_CMD_BT_SIZE 38
#define VIZIA_BT_AXIS_BT_SIZE 5
#define VIZIA_BT_SIZE 43

#define VIZIA_GV_USER_SIZE 30

#define VIZIA_GV_SLOTS_SIZE 10

struct ViziaInputStruct{
    int BT[VIZIA_BT_SIZE];
    bool BT_AVAILABLE[VIZIA_BT_SIZE];
    int BT_MAX_VALUE[VIZIA_BT_AXIS_BT_SIZE];
};

struct ViziaGameVarsStruct{
    unsigned int GAME_TIC;
    unsigned int GAME_SEED;
    unsigned int GAME_STATIC_SEED;

    unsigned int SCREEN_WIDTH;
    unsigned int SCREEN_HEIGHT;
    size_t SCREEN_PITCH;
    size_t SCREEN_SIZE;
    int SCREEN_FORMAT;

    unsigned int MAP_START_TIC;
    unsigned int MAP_TIC;

    int MAP_REWARD;

    int MAP_USER_VARS[VIZIA_GV_USER_SIZE];

    int MAP_KILLCOUNT;
    int MAP_ITEMCOUNT;
    int MAP_SECRETCOUNT;
    bool MAP_END;

    bool PLAYER_DEAD;

    int PLAYER_KILLCOUNT;
    int PLAYER_ITEMCOUNT;
    int PLAYER_SECRETCOUNT;
    int PLAYER_FRAGCOUNT; //in multi

    bool PLAYER_ON_GROUND;

    int PLAYER_HEALTH;
    int PLAYER_ARMOR;

    bool PLAYER_ATTACK_READY;
    bool PLAYER_ALTATTACK_READY;

    int PLAYER_SELECTED_WEAPON;
    int PLAYER_SELECTED_WEAPON_AMMO;

    int PLAYER_AMMO[VIZIA_GV_SLOTS_SIZE];
    int PLAYER_WEAPON[VIZIA_GV_SLOTS_SIZE];
};

#define VIZIA_SCREEN_CRCGCB 0
#define VIZIA_SCREEN_CRCGCBZB 1
#define VIZIA_SCREEN_RGB24 2
#define VIZIA_SCREEN_RGBA32 3
#define VIZIA_SCREEN_ARGB32 4
#define VIZIA_SCREEN_CBCGCR 5
#define VIZIA_SCREEN_CBCGCRZB 6
#define VIZIA_SCREEN_BGR24 7
#define VIZIA_SCREEN_BGRA32 8
#define VIZIA_SCREEN_ABGR32 9
#define VIZIA_SCREEN_GRAY8 10
#define VIZIA_SCREEN_ZBUFFER8 11
#define VIZIA_SCREEN_DOOM_256_COLORS 12

#endif
