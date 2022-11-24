/*
 * riegl_rxp.cpp
 *
 *
 * This file is part of riegl_canopy
 * This file was part of pylidar (C) Sam Gillingham and John Armston
 * Modified for riegl_tools in 2021 by Sam Gillingham and Nick Goodwin
 * Modified for riegl_canopy in 2022 by John Armston
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "pylmatrix.h"
#include "pylvector.h"

#include <riegl/scanlib.hpp>
#include <cmath>
#include <limits>

static const int nInitSize = 256*256;

/* An exception object for this module */
/* created in the init function */
struct RieglRXPState
{
    PyObject *error;
};

#define GETSTATE(m) ((struct RieglRXPState*)PyModule_GetState(m))

/* Structure for pulses */
typedef struct {
    npy_uint64 pulse_id;
    npy_uint64 timestamp;
    npy_uint8 prism_facet;
    npy_int32 scanline;
    npy_uint16 scanline_idx;
    double beam_origin_x;
    double beam_origin_y;
    double beam_origin_z;
    double beam_direction_x;
    double beam_direction_y;
    double beam_direction_z;
    npy_uint8 target_count;
} SRieglRXPPulse;

/* Structure for points */
typedef struct {
    npy_uint64 pulse_id;
    npy_uint8 target_index;
    double timestamp;
    npy_int32 deviation;
    double range;
    double reflectance;
    double amplitude;
    double x;
    double y;
    double z;
} SRieglRXPPoint;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn RieglPulseFields[] = {
    CREATE_FIELD_DEFN(SRieglRXPPulse, pulse_id, 'u'),
    CREATE_FIELD_DEFN(SRieglRXPPulse, timestamp, 'u'),
    CREATE_FIELD_DEFN(SRieglRXPPulse, prism_facet, 'u'),
    CREATE_FIELD_DEFN(SRieglRXPPulse, scanline, 'i'),
    CREATE_FIELD_DEFN(SRieglRXPPulse, scanline_idx, 'u'),
    CREATE_FIELD_DEFN(SRieglRXPPulse, beam_origin_x, 'f'),
    CREATE_FIELD_DEFN(SRieglRXPPulse, beam_origin_y, 'f'),
    CREATE_FIELD_DEFN(SRieglRXPPulse, beam_origin_z, 'f'),
    CREATE_FIELD_DEFN(SRieglRXPPulse, beam_direction_x, 'f'),
    CREATE_FIELD_DEFN(SRieglRXPPulse, beam_direction_y, 'f'),
    CREATE_FIELD_DEFN(SRieglRXPPulse, beam_direction_z, 'f'),
    CREATE_FIELD_DEFN(SRieglRXPPulse, target_count, 'u'),
    {NULL} // Sentinel
};

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn RieglPointFields[] = {
    CREATE_FIELD_DEFN(SRieglRXPPoint, pulse_id, 'u'),
    CREATE_FIELD_DEFN(SRieglRXPPoint, target_index, 'u'),
    CREATE_FIELD_DEFN(SRieglRXPPoint, timestamp, 'f'),
    CREATE_FIELD_DEFN(SRieglRXPPoint, deviation, 'i'),
    CREATE_FIELD_DEFN(SRieglRXPPoint, range, 'f'),
    CREATE_FIELD_DEFN(SRieglRXPPoint, reflectance, 'f'),
    CREATE_FIELD_DEFN(SRieglRXPPoint, amplitude, 'f'),
    CREATE_FIELD_DEFN(SRieglRXPPoint, x, 'f'),
    CREATE_FIELD_DEFN(SRieglRXPPoint, y, 'f'),
    CREATE_FIELD_DEFN(SRieglRXPPoint, z, 'f'),
    {NULL} // Sentinel
};


// Our reader class. Gets the header info and the actual data
class RieglRXPReader : public scanlib::pointcloud
{
public:
    RieglRXPReader() : scanlib::pointcloud(false),
        m_fLat(0), 
        m_fLong(0),
        m_fHeight(0),
        m_fHMSL(0),
        m_fRoll(NPY_NAN),
        m_fPitch(NPY_NAN),
        m_fYaw(NPY_NAN),
        m_beamDivergence(0),
        m_beamExitDiameter(0),
        m_thetaMin(0),
        m_thetaMax(0),
        m_phiMin(0),
        m_phiMax(0),
        m_thetaInc(0),
        m_phiInc(0),
        m_scanline(-1),
        m_scanlineIdx(0),
        m_maxScanlineIdx(0),
        m_numPulses(0),
        m_bHaveData(false),
        m_scanStarted(false),
        m_Points(nInitSize, nInitSize),
        m_Pulses(nInitSize, nInitSize)
    {

    }

    // get all the information gathered in the read 
    // as a Python dictionary.
    PyObject *getInfoDictionary()
    {
        PyObject *pDict = PyDict_New();
        PyObject *pString, *pVal;

        // we assume that the values of these variables
        // (part of the pointcloud class itself) always exist
        // as they are probably part of the preamble so if any
        // reading of the stream has been done, they should be there.
        pVal = PyLong_FromLong(num_facets);
        PyDict_SetItemString(pDict, "NUM_FACETS", pVal);
        Py_DECREF(pVal);

        pVal = PyFloat_FromDouble(group_velocity);
        PyDict_SetItemString(pDict, "GROUP_VELOCITY", pVal);
        Py_DECREF(pVal);

        pVal = PyFloat_FromDouble(unambiguous_range);
        PyDict_SetItemString(pDict, "UNAMBIGUOUS_RANGE", pVal);
        Py_DECREF(pVal);
        pString = PyUnicode_FromString(serial.c_str());

        PyDict_SetItemString(pDict, "SERIAL", pString);
        Py_DECREF(pString);
        pString = PyUnicode_FromString(type_id.c_str());

        PyDict_SetItemString(pDict, "TYPE_ID", pString);
        Py_DECREF(pString);
        pString = PyUnicode_FromString(build.c_str());

        PyDict_SetItemString(pDict, "BUILD", pString);
        Py_DECREF(pString);
        
        // now the fields that are valid if we have gathered 
        // from the 'pose' records
        if( m_bHaveData )
        {
            pVal = PyFloat_FromDouble(m_fLat);
            PyDict_SetItemString(pDict, "LATITUDE", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_fLong);
            PyDict_SetItemString(pDict, "LONGITUDE", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_fHeight);
            PyDict_SetItemString(pDict, "HEIGHT", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_fHMSL);
            PyDict_SetItemString(pDict, "HMSL", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_beamDivergence);
            PyDict_SetItemString(pDict, "BEAM_DIVERGENCE", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_beamExitDiameter);
            PyDict_SetItemString(pDict, "BEAM_EXIT_DIAMETER", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_thetaMin);
            PyDict_SetItemString(pDict, "THETA_MIN", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_thetaMax);
            PyDict_SetItemString(pDict, "THETA_MAX", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_phiMin);
            PyDict_SetItemString(pDict, "PHI_MIN", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_phiMax);
            PyDict_SetItemString(pDict, "PHI_MAX", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_thetaInc);
            PyDict_SetItemString(pDict, "THETA_INC", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_phiInc);
            PyDict_SetItemString(pDict, "PHI_INC", pVal);
            Py_DECREF(pVal);

            if( !npy_isnan(m_fRoll) )
            {
                pVal = PyFloat_FromDouble(m_fRoll);
                PyDict_SetItemString(pDict, "ROLL", pVal);
                Py_DECREF(pVal);
            }
            if( !npy_isnan(m_fPitch) )
            {
                pVal = PyFloat_FromDouble(m_fPitch);
                PyDict_SetItemString(pDict, "PITCH", pVal);
                Py_DECREF(pVal);
            }
            if( !npy_isnan(m_fYaw) )
            {
                pVal = PyFloat_FromDouble(m_fYaw);
                PyDict_SetItemString(pDict, "YAW", pVal);
                Py_DECREF(pVal);
            }

            if( !npy_isnan(m_fRoll) && !npy_isnan(m_fPitch) )
            {
                // now work out rotation matrix
                // pitch matrix
                pylidar::CMatrix<double> pitchMat(4, 4);
                pitchMat.set(0, 0, std::cos(m_fPitch));
                pitchMat.set(0, 1, 0.0);
                pitchMat.set(0, 2, std::sin(m_fPitch));
                pitchMat.set(0, 3, 0.0);
                pitchMat.set(1, 0, 0.0);
                pitchMat.set(1, 1, 1.0);
                pitchMat.set(1, 2, 0.0);
                pitchMat.set(1, 3, 0.0);
                pitchMat.set(2, 0, -std::sin(m_fPitch));
                pitchMat.set(2, 1, 0.0);
                pitchMat.set(2, 2, std::cos(m_fPitch));
                pitchMat.set(2, 3, 0.0);
                pitchMat.set(3, 0, 0.0);
                pitchMat.set(3, 1, 0.0);
                pitchMat.set(3, 2, 0.0);
                pitchMat.set(3, 3, 1.0);
            
                // roll matrix
                pylidar::CMatrix<double> rollMat(4, 4);
                rollMat.set(0, 0, 1.0);
                rollMat.set(0, 1, 0.0);
                rollMat.set(0, 2, 0.0);
                rollMat.set(0, 3, 0.0);
                rollMat.set(1, 0, 0.0);
                rollMat.set(1, 1, std::cos(m_fRoll));
                rollMat.set(1, 2, -std::sin(m_fRoll));
                rollMat.set(1, 3, 0.0);
                rollMat.set(2, 0, 0.0);
                rollMat.set(2, 1, std::sin(m_fRoll));
                rollMat.set(2, 2, std::cos(m_fRoll));
                rollMat.set(2, 3, 0.0);
                rollMat.set(3, 0, 0.0);
                rollMat.set(3, 1, 0.0);
                rollMat.set(3, 2, 0.0);
                rollMat.set(3, 3, 1.0);
            
                // yaw matrix; compass reading has been set to zero if nan
                pylidar::CMatrix<double> yawMat(4, 4);
                yawMat.set(0, 0, std::cos(m_fYaw));
                yawMat.set(0, 1, -std::sin(m_fYaw));
                yawMat.set(0, 2, 0.0);
                yawMat.set(0, 3, 0.0);
                yawMat.set(1, 0, std::sin(m_fYaw));
                yawMat.set(1, 1, std::cos(m_fYaw));
                yawMat.set(1, 2, 0.0);
                yawMat.set(1, 3, 0.0);
                yawMat.set(2, 0, 0.0);
                yawMat.set(2, 1, 0.0);
                yawMat.set(2, 2, 1.0);
                yawMat.set(2, 3, 0.0);
                yawMat.set(3, 0, 0.0);
                yawMat.set(3, 1, 0.0);
                yawMat.set(3, 2, 0.0);
                yawMat.set(3, 3, 1.0);

                // construct rotation matrix
                pylidar::CMatrix<double> tempMat = yawMat.multiply(pitchMat);
                pylidar::CMatrix<double> rotMat = tempMat.multiply(rollMat);

                pVal = (PyObject*)rotMat.getAsNumpyArray(NPY_DOUBLE);
                PyDict_SetItemString(pDict, "ROTATION_MATRIX", pVal);
                Py_DECREF(pVal);
            }

            // scanline info useful for building spatial index
            pVal = PyLong_FromLong(0);
            PyDict_SetItemString(pDict, "SCANLINE_MIN", pVal);
            Py_DECREF(pVal);

            pVal = PyLong_FromLong(m_scanline);
            PyDict_SetItemString(pDict, "SCANLINE_MAX", pVal);
            Py_DECREF(pVal);

            pVal = PyLong_FromLong(0);
            PyDict_SetItemString(pDict, "SCANLINE_IDX_MIN", pVal);
            Py_DECREF(pVal);

            pVal = PyLong_FromLong(m_maxScanlineIdx);
            PyDict_SetItemString(pDict, "SCANLINE_IDX_MAX", pVal);
            Py_DECREF(pVal);

            pVal = PyLong_FromLong(m_numPulses);
            PyDict_SetItemString(pDict, "NUMBER_OF_PULSES", pVal);
            Py_DECREF(pVal);
        }
        return pDict;
    }
    
    PyArrayObject *getPoints()
    {
        return m_Points.getNumpyArray(RieglPointFields);
    }

    PyArrayObject *getPulses()
    {
        return m_Pulses.getNumpyArray(RieglPulseFields);
    }
    
protected:
    // Not sure what the difference between the functions below
    // is but they all have more or less the same data.
    // Get scanner position and orientation packet
    void on_scanner_pose_hr_1(const scanlib::scanner_pose_hr_1<iterator_type>& arg) 
    {
        scanlib::pointcloud::on_scanner_pose_hr_1(arg);
        m_bHaveData = true;
        m_fLat = arg.LAT;
        m_fLong = arg.LON;
        m_fHeight = arg.HEIGHT;
        m_fHMSL = arg.HMSL;
        if( !npy_isnan(arg.roll))
            m_fRoll = arg.roll * pi / 180.0;
        if( !npy_isnan(arg.pitch))
            m_fPitch = arg.pitch * pi / 180.0;
        if( !npy_isnan(arg.yaw))
            m_fYaw = arg.yaw * pi / 180.0;
        else
            m_fYaw = 0; // same as original code. Correct??
    }

    void on_scanner_pose_hr(const scanlib::scanner_pose_hr<iterator_type>& arg)
    {
        scanlib::pointcloud::on_scanner_pose_hr(arg);
        m_bHaveData = true;
        m_fLat = arg.LAT;
        m_fLong = arg.LON;
        m_fHeight = arg.HEIGHT;
        m_fHMSL = arg.HMSL;
        if( !npy_isnan(arg.roll))
            m_fRoll = arg.roll * pi / 180.0;
        if( !npy_isnan(arg.pitch))
            m_fPitch = arg.pitch * pi / 180.0;
        if( !npy_isnan(arg.yaw))
            m_fYaw = arg.yaw * pi / 180.0;
        else
            m_fYaw = 0; // same as original code. Correct??
    }

    void on_scanner_pose(const scanlib::scanner_pose<iterator_type>& arg)
    {
        scanlib::pointcloud::on_scanner_pose(arg);
        m_bHaveData = true;
        m_fLat = arg.LAT;
        m_fLong = arg.LON;
        m_fHeight = arg.HEIGHT;
        m_fHMSL = arg.HMSL;
        if( !npy_isnan(arg.roll))
            m_fRoll = arg.roll * pi / 180.0;
        if( !npy_isnan(arg.pitch))
            m_fPitch = arg.pitch * pi / 180.0;
        if( !npy_isnan(arg.yaw))
            m_fYaw = arg.yaw * pi / 180.0;
        else
            m_fYaw = 0; // same as original code. Correct??
    }

    // start of a scan line going in the up direction
    void on_line_start_up(const scanlib::line_start_up<iterator_type>& arg) 
    {
        scanlib::pointcloud::on_line_start_up(arg);
        ++m_scanline;
        m_scanlineIdx = 0;
    }
    
    // start of a scan line going in the down direction
    void on_line_start_dn(const scanlib::line_start_dn<iterator_type>& arg) 
    {
        scanlib::pointcloud::on_line_start_dn(arg);
        ++m_scanline;
        m_scanlineIdx = 0;
    }

    void on_measurement_starting()
    {
        m_scanStarted = true;
    }

    void on_shot()
    {
        m_numPulses++; 
        
        SRieglRXPPulse pulse;
        pulse.pulse_id = m_numPulses;
        pulse.timestamp = time;
        pulse.prism_facet = facet;        
        pulse.beam_direction_x = beam_direction[0];
        pulse.beam_direction_y = beam_direction[1];
        pulse.beam_direction_z = beam_direction[2];
        pulse.beam_origin_x = beam_origin[0];
        pulse.beam_origin_y = beam_origin[1];
        pulse.beam_origin_z = beam_origin[2];
        pulse.scanline = m_scanline;
        pulse.scanline_idx = m_scanlineIdx; 
        pulse.target_count = 0;    

        m_scanlineIdx++;
        if( m_scanlineIdx > m_maxScanlineIdx )
        {
            m_maxScanlineIdx = m_scanlineIdx;
        }
   
        m_Pulses.push(&pulse);
    }

    // beam geometry
    void on_beam_geometry(const scanlib::beam_geometry<iterator_type>& arg) {
        scanlib::pointcloud::on_beam_geometry(arg);
        m_beamDivergence = arg.beam_divergence;
        m_beamExitDiameter = arg.beam_exit_diameter;
    }

    // scan configuration
    void on_scan_rect_fov(const scanlib::scan_rect_fov<iterator_type>& arg) {
        scanlib::pointcloud::on_scan_rect_fov(arg);
        m_thetaMin = arg.theta_min;
        m_thetaMax = arg.theta_max;
        m_phiMin = arg.phi_min;
        m_phiMax = arg.phi_max;
        m_phiInc = arg.phi_incr;
        m_thetaInc = arg.theta_incr;        
    }
    
    // overridden from pointcloud class
    void on_shot_end()
    {
        // we assume that this point will be
        // connected to the last pulse... 
        SRieglRXPPulse *pPulse = m_Pulses.getLastElement(); 
        pPulse->target_count = target_count;
 
        SRieglRXPPoint point;
        point.pulse_id = m_numPulses;

        for( scanlib::pointcloud::target_count_type target_idx = 0; target_idx < target_count; target_idx++ )
        {
            scanlib::target& current_target(targets[target_idx]);

            point.target_index = target_idx + 1;
            point.timestamp = current_target.time;
            point.deviation = current_target.deviation;
            point.reflectance = current_target.reflectance;
            point.amplitude = current_target.amplitude;

            // Get range from optical centre of scanner
            // vertex[i] = beam_origin[i] + echo_range * beam_direction[i]
            double point_range = current_target.echo_range;
            if (point_range <= std::numeric_limits<double>::epsilon()) 
            {
                current_target.vertex[0] = current_target.vertex[1] = current_target.vertex[2] = 0;
                point_range = 0;
            }
            point.range = point_range;

            point.x = current_target.vertex[0];
            point.y = current_target.vertex[1];
            point.z = current_target.vertex[2];

            m_Points.push(&point);
        }
    }


private:
    double m_fLat;
    double m_fLong;
    double m_fHeight;
    double m_fHMSL;
    double m_fRoll;
    double m_fPitch;
    double m_fYaw;
    double m_beamDivergence;
    double m_beamExitDiameter;
    double m_thetaMin;
    double m_thetaMax;
    double m_phiMin;
    double m_phiMax;
    double m_thetaInc;
    double m_phiInc;
    long m_scanline;
    long m_scanlineIdx;
    long m_maxScanlineIdx;
    long m_numPulses;
    bool m_bHaveData;
    bool m_scanStarted;
    pylidar::CVector<SRieglRXPPoint> m_Points;
    pylidar::CVector<SRieglRXPPulse> m_Pulses;
};

// reads through the whole file and returns a tuple with header
// and recarray of data.
static PyObject *rieglrxp_readFile(PyObject *self, PyObject *args)
{
    char *pszFname = NULL;

    if( !PyArg_ParseTuple(args, "s", &pszFname) )
    {
        return NULL;
    }
    
    RieglRXPReader reader;
    try
    {
        std::shared_ptr<scanlib::basic_rconnection> rc = scanlib::basic_rconnection::create(pszFname);

        scanlib::decoder_rxpmarker dec(rc);

        scanlib::buffer buf;

        for(dec.get(buf); !dec.eoi(); dec.get(buf))
        {
            reader.dispatch(buf.begin(), buf.end());
        }
    }
    catch(scanlib::scanlib_exception &e)
    {
        // raise Python exception
        PyErr_Format(GETSTATE(self)->error, "Error from Riegl lib: %s", e.what());
        return NULL;
    }   

    PyObject *pHeader = reader.getInfoDictionary();
    PyArrayObject *pPoints = reader.getPoints();
    PyArrayObject *pPulses = reader.getPulses(); 
   
    PyObject *pTuple = PyTuple_Pack(3, pHeader, pPoints, pPulses);
    Py_DECREF(pHeader);
    Py_DECREF(pPoints);
    Py_DECREF(pPulses);

    return pTuple;
}


// module methods
static PyMethodDef module_methods[] = {
    {"readFile", (PyCFunction)rieglrxp_readFile, METH_VARARGS,
        "Read the file into a numpy recarray. Pass a filename. Returns a tuple with header and array."},
    {NULL}  /* Sentinel */
};

static int rieglrxp_traverse(PyObject *m, visitproc visit, void *arg) 
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int rieglrxp_clear(PyObject *m) 
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "riegl_rxp",
        NULL,
        sizeof(struct RieglRXPState),
        module_methods,
        NULL,
        rieglrxp_traverse,
        rieglrxp_clear,
        NULL
};

PyMODINIT_FUNC PyInit_riegl_rxp(void)
{
    PyObject *pModule;
    struct RieglRXPState *state;

    /* initialize the numpy stuff */
    import_array();
    /* same for pylidar functions */
    pylidar_init();

    pModule = PyModule_Create(&moduledef);
    if( pModule == NULL )
        return NULL;

    state = GETSTATE(pModule);

    /* Create and add our exception type */
    state->error = PyErr_NewException("riegl_rxp.error", NULL, NULL);
    if( state->error == NULL )
    {
        Py_DECREF(pModule);
        return NULL;
    }
    PyModule_AddObject(pModule, "error", state->error);

    return pModule;
}

