#include <assert.h>
#include <cstdlib>
#include <fmt/core.h>
#include "hdf5.h"
#include <iostream>
#include <string.h>
#include "util.h"

#include "h5_reader.h"

using namespace std;


/*
  Opens an h5 file for reading, assuming it contains standard radio
  telescope input from one of the known telescopes. If the data is an
  unexpected size or shape we should be conservative and exit.
 */
H5Reader::H5Reader(const string& filename) : FilterbankFileReader(filename) {
  file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file == H5I_INVALID_HID) {
    fatal("could not open file for reading:", filename);
  }
  dataset = H5Dopen2(file, "data", H5P_DEFAULT);
  if (dataset == H5I_INVALID_HID) {
    fatal("could not open dataset");
  }
  if (!H5Tequal(H5Dget_type(dataset), H5T_NATIVE_FLOAT)) {
    fatal("dataset is not float");
  }
    
  fch1 = getDoubleAttr("fch1");
  foff = getDoubleAttr("foff");
  tstart = getDoubleAttr("tstart");
  tsamp = getDoubleAttr("tsamp");
  src_dej = getDoubleAttr("src_dej");
  src_raj = getDoubleAttr("src_raj");
  source_name = getStringAttr("source_name");
  
  dataspace = H5Dget_space(dataset);
  if (dataspace == H5I_INVALID_HID) {
    fatal("could not open dataspace");
  }
  if (H5Sget_simple_extent_ndims(dataspace) != 3) {
    fatal("data is not three-dimensional");
  }
  hsize_t dims[3];
  H5Sget_simple_extent_dims(dataspace, dims, NULL);
  if (dims[1] != 1) {
    fatal(fmt::format("unexpected second dimension: {}", dims[1]));
  }
  num_timesteps = dims[0];
  num_channels = dims[2];
  
  telescope_id = getLongAttr("telescope_id");

  if (attrExists("nfpc")) {
    coarse_channel_size = getLongAttr("nfpc");
  }

  inferMetadata();
}

double H5Reader::getDoubleAttr(const string& name) const {
  double output;
  auto attr = H5Aopen(dataset, name.c_str(), H5P_DEFAULT);
  if (attr == H5I_INVALID_HID) {
    fatal("could not access attr", name);
  }
  if (H5Aread(attr, H5T_NATIVE_DOUBLE, &output) < 0) {
    fatal("attr could not be read as double:", name);
  }
  H5Aclose(attr);
  return output;
}

long H5Reader::getLongAttr(const string& name) const {
  long output;
  auto attr = H5Aopen(dataset, name.c_str(), H5P_DEFAULT);
  if (attr == H5I_INVALID_HID) {
    fatal("could not access attr", name);
  }
  if (H5Aread(attr, H5T_NATIVE_LONG, &output) < 0) {
    fatal("attr could not be read as long:", name);
  }
  H5Aclose(attr);
  return output;  
}

bool H5Reader::attrExists(const string& name) const {
  auto answer = H5Aexists(dataset, name.c_str());
  if (answer < 0) {
    fatal("existence check failed for attr", name);
  }
  return answer > 0;
}

/*
  This assumes the string is stored as variable-length UTF8.
  I'm not sure what if any type conversion the library will do, and
  historically we do not store attributes with consistent string
  subtypes, so when we run into attributes with different formats we
  might have to improve this method.
 */
string H5Reader::getStringAttr(const string& name) const {
  auto attr = H5Aopen(dataset, name.c_str(), H5P_DEFAULT);
  if (attr == H5I_INVALID_HID) {
    fatal("could not access attr", name);
  }

  // Check the attribute's character type
  auto attr_type = H5Aget_type(attr);
  auto cset = H5Tget_cset(attr_type);
  if (cset < 0) {
    fatal("H5Tget_cset failed");
  }
  
  // Create mem_type for variable-length string of our character type
  auto mem_type = H5Tcopy(H5T_C_S1);
  if (H5Tset_size(mem_type, H5T_VARIABLE) < 0) {
    fatal("H5Tset_size failed");
  }
  if (H5Tset_strpad(mem_type, H5T_STR_NULLTERM) < 0) {
    fatal("H5Tset_strpad failed");
  }
  if (H5Tset_cset(mem_type, cset) < 0) {
    fatal("H5Tset_cset failed");
  }

  // We need to add one ourselves for a null
  auto storage_size = H5Aget_storage_size(attr) + 1;
  char* buffer = (char*)malloc(storage_size * sizeof(char));
  memset(buffer, 0, storage_size);

  // The API for variable-length and fixed-length attributes is
  // different, so first we determine which one we are reading
  bool variable_length = H5Tequal(mem_type, attr_type);
  if (variable_length) {
    if (H5Aread(attr, mem_type, &buffer) < 0) {
      fatal("variable-length H5Aread failed for", name);
    }
  } else {
    auto fixed_type = H5Tcopy(attr_type);
    if (H5Aread(attr, fixed_type, buffer) < 0) {
      fatal("fixed-length H5Aread failed for", name);
    }
    H5Tclose(fixed_type);
  }
  
  string output(buffer);
  
  free(buffer);
  H5Aclose(attr);
  H5Tclose(attr_type);
  H5Tclose(mem_type);

  return output;
}

/*
  Loads the data in row-major order.

  If the buffer has extra space beyond that needed to load the coarse channel, we zero it out.

  This also corrects for the DC spike, if needed.
*/
void H5Reader::loadCoarseChannel(int i, FilterbankBuffer* buffer) const {
  assert(num_timesteps <= buffer->num_timesteps);
  assert(coarse_channel_size == buffer->num_channels);
  
  // Select a hyperslab containing just the coarse channel we want
  const hsize_t offset[3] = {0, 0, unsigned(i * coarse_channel_size)};
  const hsize_t coarse_channel_dim[3] = {unsigned(num_timesteps), 1,
                                         unsigned(coarse_channel_size)};
  if (H5Sselect_hyperslab(dataspace, H5S_SELECT_SET,
                          offset, NULL, coarse_channel_dim, NULL) < 0) {
    fatal("failed to select coarse channel hyperslab");
  }

  // Define a memory dataspace
  hid_t memspace = H5Screate_simple(3, coarse_channel_dim, NULL);
  if (memspace == H5I_INVALID_HID) {
    fatal("failed to create memspace");
  }
    
  // Copy from dataspace to memspace
  if (H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT,
              buffer->sg_data) < 0) {    
    fatal("h5 read failed. make sure that plugin files are in the plugin directory:",
          H5_DEFAULT_PLUGINDIR);
  }
    
  H5Sclose(memspace);

  if (num_timesteps < buffer->num_timesteps) {
    // Zero out the extra buffer space
    int num_floats_loaded = num_timesteps * coarse_channel_size;
    int num_zeros_needed = (buffer->num_timesteps - num_timesteps) * coarse_channel_size;
    memset(buffer->sg_data + num_floats_loaded, 0, num_zeros_needed * sizeof(float));
  }

  if (has_dc_spike) {
    // Remove the DC spike by making it the average of the adjacent columns
    int mid = coarse_channel_size / 2;
    for (int row_index = 0; row_index < num_timesteps; ++row_index) {
      float* row = buffer->sg_data + row_index * coarse_channel_size;
      row[mid] = (row[mid - 1] + row[mid + 1]) / 2.0;
    }
  }
}
  
H5Reader::~H5Reader() {
  H5Sclose(dataspace);
  H5Dclose(dataset);
  H5Fclose(file);
}

