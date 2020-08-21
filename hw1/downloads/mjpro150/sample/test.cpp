//---------------------------------//
//  This file is part of MuJoCo    //
//  Written by Emo Todorov         //
//  Copyright (C) 2017 Roboti LLC  //
//---------------------------------//


#include "mujoco.h"
#include "mjxmacro.h"
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <string>
#include <chrono>


using namespace std;

// timer
double gettm(void)
{
    static chrono::system_clock::time_point _start = chrono::system_clock::now();
    chrono::duration<double> elapsed = chrono::system_clock::now() - _start;
    return elapsed.count();
}


// help
const char helpstring[] = 
    "\n Usage:  test modelfile option [duration]\n"
    "   option can contain: 'x'- xml save/load, 's'- speed\n"
    "   if 's' is included, the simulation runs for the specified duration in seconds\n\n"
    " Example:  test model.xml xs 10\n";


// deallocate and print message
int finish(const char* msg = 0, mjModel* m = 0, mjData* d = 0)
{
    // deallocated everything
    if( d )
        mj_deleteData(d);
    if( m )
        mj_deleteModel(m);
    mj_deactivate();

    // print message
    if( msg )
        printf("%s\n", msg);

    return 0;
}


// return absolute difference if it is below 1, relative difference otherwise
static mjtNum _compare(mjtNum val1, mjtNum val2)
{
    mjtNum magnitude = mju_max(mju_abs(val1), mju_abs(val2));

    if( magnitude>1.0 )
        return mju_abs(val1-val2) / magnitude; 
    else
        return mju_abs(val1-val2);
}


// compare two models, return largest difference and field name
mjtNum compareModel(const mjModel* m1, const mjModel* m2, char* field)
{
    int r, c;
    mjtNum dif, maxdif = 0.0;

    // define symbols corresponding to number of columns (needed in MJMODEL_POINTERS)
    int nq              = m1->nq;
    int nv              = m1->nv;
    int na              = m1->na;
    int nuser_body      = m1->nuser_body;
    int nuser_jnt       = m1->nuser_jnt;
    int nuser_geom      = m1->nuser_geom;
    int nuser_site      = m1->nuser_site;
    int nuser_cam       = m1->nuser_cam;
    int nuser_tendon    = m1->nuser_tendon;
    int nuser_actuator  = m1->nuser_actuator;
    int nuser_sensor    = m1->nuser_sensor;

    // compare ints
    #define X(name) if(m1->name!=m2->name) {strcpy(field, #name); return 1.0;}

        MJMODEL_INTS
    #undef X

    // compare arrays
    #define X(type, name, nr, nc)                               \
        for( r=0; r<m1->nr; r++ )                               \
        for( c=0; c<nc; c++ ) {                                 \
            dif = _compare(m1->name[r*nc+c], m2->name[r*nc+c]); \
            if(dif>maxdif) {maxdif=dif; strcpy(field, #name);} }

        MJMODEL_POINTERS
    #undef X

    // compare scalars in mjOption
    #define X(type, name)                                       \
        dif = _compare(m1->opt.name, m2->opt.name);             \
        if(dif>maxdif) {maxdif=dif; strcpy(field, #name);}

        MJOPTION_SCALARS
    #undef X

    // compare arrays in mjOption
    #define X(name, n)                                          \
        for( c=0; c<n; c++ ) {                                  \
            dif = _compare(m1->opt.name[c], m2->opt.name[c]);   \
            if(dif>maxdif) {maxdif=dif; strcpy(field, #name);} }

        MJOPTION_VECTORS
    #undef X

    // mjVisual and mjStatistics ignored for now

    return maxdif;
}



// main function
int main(int argc, const char** argv)
{
    // print help if arguments are missing
    if( argc<3 )
        return finish(helpstring);

    // activate MuJoCo Pro license (this must be *your* activation key)
    mj_activate("mjkey.txt");

    // get filename, determine file type
    std::string filename(argv[1]);
    bool binary = (filename.find(".mjb")!=std::string::npos);

    // load model
    mjModel* m = 0;
    char error[1000] = "Could not load binary model";
    if( binary )
        m = mj_loadModel(argv[1], 0);
    else
        m = mj_loadXML(argv[1], 0, error, 1000);
    if( !m )
        return finish(error);

    // make data
    mjData* d = mj_makeData(m);
    if( !d )
        return finish("Could not allocate mjData", m);

    // get option
    std::string option(argv[2]);

    // save/load test
    if( option.find_first_of('x')!=std::string::npos )
    {
        // require xml model
        if( binary )
            return finish("XML model is required for save/load test", m, d);

        // prepare temp filename in the same directory as original (for asset loading)
        std::string tempfile;
        size_t lastpath = filename.find_last_of("/\\");
        if( lastpath==std::string::npos )
            tempfile = "_tempfile_.xml";
        else
            tempfile = filename.substr(0, lastpath+1) + "_tempfile_.xml";

        // save
        if( !mj_saveLastXML(tempfile.c_str(), m, error, 1000) )
            return finish(error, m, d);

        // load back
        mjModel* mtemp = mj_loadXML(tempfile.c_str(), 0, error, 100);
        if( !mtemp )
            return finish(error, m, d);

        // compare
        char field[500] = "";
        mjtNum result = compareModel(m, mtemp, field);
        printf("\nComparison of original and saved model\n");
        printf(" Max difference : %.3g\n", result);
        printf(" Field name     : %s\n", field);

        // delete temp model and file
        mj_deleteModel(mtemp);
        remove(tempfile.c_str());
    }

    // speed test
    if( option.find_first_of('s')!=std::string::npos )
    {
        // require duration
        if( argc<4 )
            return finish("Duration argument is required for speed test", m, d);

        // read duration
        double duration = 0;
        if( sscanf(argv[3], "%lf", &duration)!=1 || duration<=0 )
            return finish("Invalid duration argument", m, d);

        // time simulation
        int steps = 0, contacts = 0, constraints = 0;
        double printfraction = 0.1;
        printf("\nSimulation ");
        double start = gettm();
        while( d->time<duration )
        {
            // advance simulation
            mj_step(m, d);

            // accumulate statistics
            steps++;
            contacts += d->ncon;
            constraints += d->nefc;

            // print '.' every 10% of duration
            if( d->time >= duration*printfraction )
            {
                printf(".");
                printfraction += 0.1;
            }
        }
        double end = gettm();

        // print results
        printf("\n Simulation time      : %.2f s\n", end-start);
        printf(" Realtime factor      : %.2f x\n", duration/mjMAX(1E-10,(end-start)));
        printf(" Time per step        : %.3f ms\n", 1000.0*(end-start)/mjMAX(1,steps));
        printf(" Contacts per step    : %d\n", contacts/mjMAX(1,steps));
        printf(" Constraints per step : %d\n", constraints/mjMAX(1,steps));
        printf(" Degrees of freedom   : %d\n\n", m->nv);
    }

    // finalize
    return finish();
}
