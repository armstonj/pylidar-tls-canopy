/*
 * riegl_rdb.cpp
 *
 * This file is part of riegl-tls-canopy
 * This file was part of pylidar (C) Sam Gillingham
 * Modified for riegl_tools in 2021 by Sam Gillingham
 * Modified for pylidar-tls-canopy in 2023 by John Armston
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
#include <cmath>
#include <memory>
#include <unordered_map>
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "pylvector.h"

#include "riegl/rdb.h"
#include "riegl/rdb/default/attributes.h"

/* An exception object for this module */
/* created in the init function */
struct RieglRDBState
{
    PyObject *error;
};

#define GETSTATE(m) ((struct RieglRDBState*)PyModule_GetState(m))
static RieglRDBState* GETSTATE_FC();

static const int nInitSize = 256*256;

#ifdef _MSC_VER
// not available with MSVC
// see https://bugs.libssh.org/T112
char *strndup(const char *s, size_t n)
{
    char *x = NULL;

    if (n + 1 < n) {
        return NULL;
    }

    x = (char*)malloc(n + 1);
    if (x == NULL) {
        return NULL;
    }

    memcpy(x, s, n);
    x[n] = '\0';

    return x;
}

#endif

/* Structure for points */
typedef struct {
    npy_uint8 target_index;
    npy_uint8 target_count;
    double timestamp;
    npy_int32 deviation;
    npy_uint16 classification;
    double range;
    double reflectance;
    double amplitude;
    double x;
    double y;
    double z;
    npy_int32 scanline;
    npy_int32 scanline_idx;
    npy_uint64 riegl_id; // riegl.id
} SRieglRDBPoint;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn RieglPointFields[] = {
    CREATE_FIELD_DEFN(SRieglRDBPoint, target_index, 'u'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, target_count, 'u'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, timestamp, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, deviation, 'i'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, classification, 'u'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, range, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, reflectance, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, amplitude, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, x, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, y, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, z, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, scanline, 'i'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, scanline_idx, 'i'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, riegl_id, 'u'),
    {NULL} // Sentinel
};

// buffer that we read the data in before copying to point/pulse structures
// NOTE: any changes must be reflected in RiegRDBReader::resetBuffers()
typedef struct
{
    // points
    npy_uint64 id; // riegl.id
    double timestamp; // riegl.timestamp 
    npy_int32 deviation; // riegl.deviation
    npy_uint16 classification; // riegl.class
    double reflectance; // riegl.reflectance
    double amplitude; // riegl.amplitude
    double xyz[3]; // riegl.xyz
    
    // pulses
    npy_int32 scan_line_index;  // scanline
    npy_int32 shot_index_line; // scanline_idx
    
    // info for attributing points to pulses
    npy_uint8 target_index; // riegl.target_index
    npy_uint8 target_count; // riegl.target_count

} RieglRDBBuffer;

// Returns false and sets Python exception if error code set
static bool CheckRDBResult(RDBResult code, RDBContext *pContext);

// For use in PyRieglRDBFile_* functions, return ret on error
#define CHECKRESULT_FILE(code, ret) if( !CheckRDBResult(code, pContext) ) return ret;

// For use in RiegRDBReader, return ret on error
#define CHECKRESULT_READER(code, ret) if( !CheckRDBResult(code, pContext) ) return ret;
// for automating the binding process
#define CHECKBIND_READER(attribute, dataType, buffer) if( !CheckRDBResult( \
                    rdb_pointcloud_query_select_bind( \
                        pContext, pQuerySelect, attribute, dataType, buffer, \
                        sizeof(RieglRDBBuffer)), pContext) ) return NULL;
// try one name, then the other
#define CHECKBIND_READER2(attribute1, attribute2, dataType, buffer) if( \
        (rdb_pointcloud_query_select_bind(pContext, pQuerySelect, attribute1, dataType, buffer, \
        sizeof(RieglRDBBuffer)) != RDB_SUCCESS) && !CheckRDBResult( \
            rdb_pointcloud_query_select_bind( \
                        pContext, pQuerySelect, attribute2, dataType, buffer, \
                        sizeof(RieglRDBBuffer)), pContext) ) return NULL;

// Processes psz until a '.' is found
// and returns string as an int. 
// Updates psz to point to first char after the '.'
static int GetAVersionPart(RDBString *psz)
{
RDBString pCurr = *psz;

    while( *pCurr != '.' )
    {
        pCurr++;
    }
    
    char *pTemp = strndup(*psz, pCurr - (*psz));
    int nResult = atoi(pTemp);
    free(pTemp);
    
    pCurr++; // go past dot
    *psz = pCurr; // store back to caller
    
    return nResult;
}

// For qsort - does comparison on rieglId field
int rieglId_cmpfunc(const void * a, const void * b)
{
    const SRieglRDBPoint *pa = static_cast<const SRieglRDBPoint*>(a);
    const SRieglRDBPoint *pb = static_cast<const SRieglRDBPoint*>(b);
   
    // rieglId is uint64 so not a great idea to convert to int
    // by subracting and returning the result
    if( pb->riegl_id < pa->riegl_id )
    {
        return 1;
    }
    else if( pa->riegl_id == pb->riegl_id)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

// our main function
static PyObject *rieglrdb_readFile(PyObject *self, PyObject *args)
{
    char *pszFname = NULL;
    int bSort = 0;
    RDBContext *pContext = NULL;
    RDBPointcloud *pPointCloud = NULL;
    RDBPointcloudQuerySelect *pQuerySelect = NULL;

    if( !PyArg_ParseTuple(args, "s|p", &pszFname, &bSort) )
    {
        return NULL;
    }

    // get context
    // TODO: log level and log path as an option?
    CHECKRESULT_FILE(rdb_context_new(&pContext, "", ""), NULL)
    
    // check we are running against the same version of the library we were compiled
    // against
    RDBString pszVersionString;
    CHECKRESULT_FILE(rdb_library_version(pContext, &pszVersionString), NULL)
    
    RDBString psz = pszVersionString;
    int nMajor = GetAVersionPart(&psz);
    int nMinor = GetAVersionPart(&psz);
    if( (nMajor != RIEGL_RDB_MAJOR) || (nMinor != RIEGL_RDB_MINOR) )
    {
        // raise Python exception
        PyErr_Format(GETSTATE_FC()->error, "Mismatched libraries - RDB lib differs in version "
            "Was compiled against version %d.%d. Now running with %d.%d (version string: '%s')\n", 
            RIEGL_RDB_MAJOR, RIEGL_RDB_MINOR, nMajor, nMinor, pszVersionString);
        return NULL;
    }
    
    // open file
    // need a RDBPointcloudOpenSettings TODO: set from options
    RDBPointcloudOpenSettings *pSettings;
    CHECKRESULT_FILE(rdb_pointcloud_open_settings_new(pContext, &pSettings), NULL)
    
    CHECKRESULT_FILE(rdb_pointcloud_new(pContext, &pPointCloud), NULL)
    
    CHECKRESULT_FILE(rdb_pointcloud_open(pContext, pPointCloud,
                    pszFname, pSettings), NULL)
    
    // read in header ("metadata" in rdblib speak)
    PyObject *pHeader = PyDict_New();
    uint32_t acount;
    RDBString alist;
    CHECKRESULT_FILE(rdb_pointcloud_meta_data_list(pContext, pPointCloud,
                        &acount, &alist), NULL)
    RDBString value;
    for( uint32_t i = 0; i < acount; i++ )
    {
        CHECKRESULT_FILE(rdb_pointcloud_meta_data_get(pContext, pPointCloud,
                        alist, &value), NULL)
        // JSON decoding happens in Python
        PyObject *pString = PyUnicode_FromString(value);
        PyDict_SetItemString(pHeader, alist, pString);
        Py_DECREF(pString);
        /*fprintf(stderr, "%d %s %s\n", i, list, value);*/
        alist = alist + strlen(alist) + 1;
    }

    ///// Now read file
    CHECKRESULT_FILE(rdb_pointcloud_query_select_new(pContext, 
                pPointCloud,
                0, // all nodes apparently according to querySelect.cpp
                "",
                &pQuerySelect), NULL)

    RieglRDBBuffer *pBuffer = new RieglRDBBuffer[nInitSize];
    pylidar::CVector<SRieglRDBPoint> points(nInitSize, nInitSize);

    bool bEOF = false;
    while( !bEOF )
    {
        CHECKBIND_READER(RDB_RIEGL_ID.name, RDBDataTypeUINT64, &pBuffer[0].id)
        CHECKBIND_READER(RDB_RIEGL_TIMESTAMP.name, RDBDataTypeDOUBLE, &pBuffer[0].timestamp)
        CHECKBIND_READER(RDB_RIEGL_DEVIATION.name, RDBDataTypeINT32, &pBuffer[0].deviation)
        CHECKBIND_READER(RDB_RIEGL_CLASS.name, RDBDataTypeUINT16, &pBuffer[0].classification)
        CHECKBIND_READER(RDB_RIEGL_REFLECTANCE.name, RDBDataTypeDOUBLE, &pBuffer[0].reflectance)
        CHECKBIND_READER(RDB_RIEGL_AMPLITUDE.name, RDBDataTypeDOUBLE, &pBuffer[0].amplitude)
        CHECKBIND_READER(RDB_RIEGL_XYZ.name, RDBDataTypeDOUBLE, &pBuffer[0].xyz)
        
        // riegl.row and riegl.column are used on older files (don't appear to be documented, but are in there)
        // newer files use RDB_RIEGL_SCAN_LINE_INDEX / RDB_RIEGL_SHOT_INDEX_LINE
        CHECKBIND_READER(RDB_RIEGL_SCAN_LINE_INDEX.name, RDBDataTypeINT32, &pBuffer[0].scan_line_index);
        CHECKBIND_READER(RDB_RIEGL_SHOT_INDEX_LINE.name, RDBDataTypeINT32, &pBuffer[0].shot_index_line);
        
        CHECKBIND_READER(RDB_RIEGL_TARGET_INDEX.name, RDBDataTypeUINT8, &pBuffer[0].target_index)
        CHECKBIND_READER(RDB_RIEGL_TARGET_COUNT.name, RDBDataTypeUINT8, &pBuffer[0].target_count)
        
        uint32_t processed = 0;
        if( rdb_pointcloud_query_select_next(
                pContext, pQuerySelect, nInitSize,
                &processed) == 0 )
        {
            // 0 means 'end of file' apparently
            bEOF = true;
        }
        if( processed == 0 )
        {
            bEOF = true;
        }

        for( uint32_t nPointIdx = 0; nPointIdx < processed; nPointIdx++ )
        {
            SRieglRDBPoint point;
            RieglRDBBuffer *pCurrEl = &pBuffer[nPointIdx];

            // create point
            // pylidar used to do abs() of this, we don't, not sure
            // point.scanline = pCurrEl->row;
            // point.scanline_Idx = pCurrEl->column;

            // convert to ns
            point.timestamp = pCurrEl->timestamp;
            point.deviation = pCurrEl->deviation;
            point.classification = pCurrEl->classification;
            point.range = std::sqrt(std::pow(pCurrEl->xyz[0], 2) + 
                            std::pow(pCurrEl->xyz[1], 2) + 
                            std::pow(pCurrEl->xyz[2], 2));
            point.reflectance = pCurrEl->reflectance;
            point.amplitude = pCurrEl->amplitude;
            point.x = pCurrEl->xyz[0];
            point.y = pCurrEl->xyz[1];
            point.z = pCurrEl->xyz[2];
            point.scanline = pCurrEl->scan_line_index;
            point.scanline_idx = pCurrEl->shot_index_line;
            point.target_index = pCurrEl->target_index;
            point.target_count = pCurrEl->target_count;
            point.riegl_id = pCurrEl->id;

            points.push(&point);
        }
    }

    delete[] pBuffer;
    rdb_pointcloud_query_select_delete(pContext, &pQuerySelect);
    rdb_pointcloud_delete(pContext, &pPointCloud);
    rdb_context_delete(&pContext);
    
    if( bSort )
    {
        points.sort(rieglId_cmpfunc);
    }

    PyArrayObject *pPoints = points.getNumpyArray(RieglPointFields);
    PyObject *pTuple = PyTuple_Pack(2, pHeader, pPoints);
    Py_DECREF(pHeader);
    Py_DECREF(pPoints);

    return pTuple;
}

static PyObject *rieglrdb_readHeader(PyObject *self, PyObject *args)
{
    char *pszFname = NULL;
    RDBContext *pContext = NULL;
    RDBPointcloud *pPointCloud = NULL;

    if( !PyArg_ParseTuple(args, "s", &pszFname) )
    {
        return NULL;
    }

    // get context
    // TODO: log level and log path as an option?
    CHECKRESULT_FILE(rdb_context_new(&pContext, "", ""), NULL)
    
    // check we are running against the same version of the library we were compiled
    // against
    RDBString pszVersionString;
    CHECKRESULT_FILE(rdb_library_version(pContext, &pszVersionString), NULL)
    
    RDBString psz = pszVersionString;
    int nMajor = GetAVersionPart(&psz);
    int nMinor = GetAVersionPart(&psz);
    if( (nMajor != RIEGL_RDB_MAJOR) || (nMinor != RIEGL_RDB_MINOR) )
    {
        // raise Python exception
        PyErr_Format(GETSTATE_FC()->error, "Mismatched libraries - RDB lib differs in version "
            "Was compiled against version %d.%d. Now running with %d.%d (version string: '%s')\n", 
            RIEGL_RDB_MAJOR, RIEGL_RDB_MINOR, nMajor, nMinor, pszVersionString);
        return NULL;
    }
    
    // open file
    // need a RDBPointcloudOpenSettings TODO: set from options
    RDBPointcloudOpenSettings *pSettings;
    CHECKRESULT_FILE(rdb_pointcloud_open_settings_new(pContext, &pSettings), NULL)
    
    CHECKRESULT_FILE(rdb_pointcloud_new(pContext, &pPointCloud), NULL)
    
    CHECKRESULT_FILE(rdb_pointcloud_open(pContext, pPointCloud,
                    pszFname, pSettings), NULL)
    
    // read in header ("metadata" in rdblib speak)
    PyObject *pHeader = PyDict_New();
    uint32_t acount;
    RDBString alist;
    CHECKRESULT_FILE(rdb_pointcloud_meta_data_list(pContext, pPointCloud,
                        &acount, &alist), NULL)
    RDBString value;
    for( uint32_t i = 0; i < acount; i++ )
    {
        CHECKRESULT_FILE(rdb_pointcloud_meta_data_get(pContext, pPointCloud,
                        alist, &value), NULL)
        // JSON decoding happens in Python
        PyObject *pString = PyUnicode_FromString(value);
        PyDict_SetItemString(pHeader, alist, pString);
        Py_DECREF(pString);
        /*fprintf(stderr, "%d %s %s\n", i, list, value);*/
        alist = alist + strlen(alist) + 1;
    }
    
    rdb_pointcloud_delete(pContext, &pPointCloud);
    rdb_context_delete(&pContext);

    return pHeader;
}

// module methods
static PyMethodDef module_methods[] = {
    {"readFile", (PyCFunction)rieglrdb_readFile, METH_VARARGS,
        "Read the file into a numpy recarray. Pass a filename (and optional sort flag - default is True). "
        "Returns a tuple with header and array."},
    {"readHeader", (PyCFunction)rieglrdb_readHeader, METH_VARARGS,
        "Return the header for the file"},
    {NULL}  /* Sentinel */
};

static int rieglrdb_traverse(PyObject *m, visitproc visit, void *arg) 
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int rieglrdb_clear(PyObject *m) 
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "riegl_rdb",
        NULL,
        sizeof(struct RieglRDBState),
        module_methods,
        NULL,
        rieglrdb_traverse,
        rieglrdb_clear,
        NULL
};

static RieglRDBState* GETSTATE_FC()
{
    return GETSTATE(PyState_FindModule(&moduledef));
}

// Returns false and sets Python exception if error code set
static bool CheckRDBResult(RDBResult code, RDBContext *pContext)
{
    if( code != RDB_SUCCESS )
    {
        int32_t   errorCode(0);
        RDBString text("Unable to create context");
        RDBString details("");
    
        if( pContext != NULL )
        {
            rdb_context_get_last_error(pContext, &errorCode, &text, &details);
        }
        
        PyErr_Format(GETSTATE_FC()->error, "Error from RDBLib: %s. Details: %s\n",
               text, details); 
        return false;
    }
    
    return true;
}

PyMODINIT_FUNC PyInit_riegl_rdb(void)
{
    PyObject *pModule;
    struct RieglRDBState *state;

    /* initialize the numpy stuff */
    import_array();
    /* same for pylidar functions */
    pylidar_init();

    pModule = PyModule_Create(&moduledef);
    if( pModule == NULL )
        return NULL;

    state = GETSTATE(pModule);

    /* Create and add our exception type */
    state->error = PyErr_NewException("riegl_rdb.error", NULL, NULL);
    if( state->error == NULL )
    {
        Py_DECREF(pModule);
        return NULL;
    }
    PyModule_AddObject(pModule, "error", state->error);

    return pModule;
}
