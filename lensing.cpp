#include <png++/png.hpp>
#include <bits/stdc++.h>
#include <omp.h>

std::string filename = "background.png";

int nx, ny, mx, my;

using imtype = png::image<png::rgb_pixel>;
imtype image;
imtype out_image;

float rbh[3];
float Mbh = 50.0f;
constexpr float Rbh = 1.0f;
constexpr float base_dt = 0.1f;
constexpr float thresh = 1.0e3;
constexpr int max_ite = 100000;
constexpr float eta = 1.0e-2;
constexpr float pi = acos(-1.0f);

bool spherical_projection = false;

bool debug = false;
int trace_x = 300;
int trace_y = 450;

constexpr int base_nx = 800;
constexpr int base_ny = 800;

constexpr int CR_OK  = 0;
constexpr int CR_ITE = 1; // Too many iterations
constexpr int CR_OUT = 2; // Out of the box
constexpr int CR_ABS = 3; // Entered the BH

int cast_ray(int x, int y, int &ox, int &oy, bool trace) {
  float r[3] {(float)x, (float)y, 0.0};
  float v[3] {0, 0, 1.0};
  int ite = 0;
  float dv[3];
  float d, d3;
  float F[3] {0.0, 0.0, 0.0}; 
  float dt = base_dt;
  
  std::ofstream f_out;

  if (trace)
    f_out.open("trace.dat");

  d = 0.0f;
  
  for (int i=0; i < 3; ++i) {
    dv[i] = rbh[i] - r[i];
    d += dv[i]*dv[i];
  }
  d = sqrt(d);
  
  while (r[2] < thresh) {
    if (trace)
      f_out << r[0] << " " << r[1] << " " << r[2] << " " << dt << std::endl;
    
    ite++;

    if (d < Rbh)
      return CR_ABS;

    if (ite > max_ite)
      return CR_ITE;

    d3 = d*d*d;
       d = 0;
    for (int i=0; i < 3; ++i) {
      dv[i] = rbh[i] - r[i];
      d += dv[i]*dv[i];
    }
    d = sqrt(d);
 
    for (int i=0; i < 3; ++i)
      v[i] += F[i] * dt * 0.5f;

    // Updating timestep
    dt = eta * sqrt(d3/Mbh);
    
    for (int i=0; i < 3; ++i) {
      F[i] = Mbh * dv[i] / d3;
      v[i] += F[i] * dt * 0.5f;
      r[i] += v[i] * dt;
    }

    d = 0;
    for (int i=0; i < 3; ++i) {
      dv[i] = rbh[i] - r[i];
      d += dv[i]*dv[i];
    }
    d = sqrt(d);
  }

  // Todo : interpolate colors here ?
  if (spherical_projection) {
    float nr[3];
    nr[0] = (rbh[0]-r[0]) / d;
    nr[1] = (rbh[1]-r[1]) / d;
    nr[2] = (rbh[2]-r[2]) / d;
    
    ox = mx * (0.5 + atan2(nr[2], nr[1]) / (2.0 * pi));
    oy = my * (0.5 + asin(nr[0]) / pi);

    if (oy < 0) {
      std::cerr << "ARG : " << nr[0] << " " << nr[1] << " " << nr[2] << " " << ox << " " << oy << std::endl;
    }
  }
  else {
    ox = int(r[0]);
    oy = int(r[1]);

    // Testing on ox/oy because of potential overflows
    if (ox < 0.0 || ox >= nx || oy < 0.0 || oy >= ny)
      return CR_OUT;
  }

  if (trace)
    f_out.close();
  
  return CR_OK;
}

int main(int argc, char** argv) {
  // Reading input
  image.read(filename);

  if (!spherical_projection) {
    ny = image.get_height();
    nx = image.get_width();
  }
  else {
    ny = base_ny;
    nx = base_nx;
  }
  
  mx = image.get_width();
  my = image.get_height();
  
  out_image = imtype(nx, ny);

  // How many threads are running this ?
#pragma omp parallel
  {
#pragma omp master
    std::cerr << "Running with " << omp_get_num_threads() << " threads" << std::endl;
  }

  float bhx = 0.5f;
  float bhy = 0.5f;
  float bhz = 0.5f;

  int aid = 1;
  while (aid < argc) {
    std::string arg(argv[aid]);
    
    if (arg == "--x") {
      bhx = strtod(argv[aid+1], nullptr);
      aid++;
    }
    else if (arg == "--y") {
      bhy = strtod(argv[aid+1], nullptr);
      aid++;
    }
    else if (arg == "--z") {
      bhz = strtod(argv[aid+1], nullptr);
      aid++;
    }
    else if (arg == "--M") {
      Mbh = strtod(argv[aid+1], nullptr);
      aid++;
    }
    else if (arg == "--debug")
      debug = true;
    
    aid++;
  }

  rbh[0] = nx * bhx;
  rbh[1] = ny * bhy;
  rbh[2] = thresh * bhz;

  png::rgb_pixel color_out, color_ite, color_abs;

  // If debug mode, the invalid pixels have different colors depending on the case !
  if (debug) {
    color_out = png::rgb_pixel(255, 0, 0);
    color_ite = png::rgb_pixel(0, 255, 0);
    color_abs = png::rgb_pixel(0, 0, 255);
  }
  else { // All invalid pixels are black
    color_out = png::rgb_pixel(0, 0, 0);
    color_ite = png::rgb_pixel(0, 0, 0);
    color_abs = png::rgb_pixel(0, 0, 0);
  }


  for (int i=0; i < ny; ++i) {
    std::cerr << "\rRow " << i+1 << " / " << ny;
    std::cerr.flush();
    
#pragma omp parallel for shared(i, image, out_image)
    for (int j=0; j < nx; ++j) {
      int ox, oy;
      int cr_code;
      bool trace = (j == trace_y && i == trace_x);
      cr_code = cast_ray(j, i, ox, oy, trace);

      switch(cr_code) {
      case CR_OK:  out_image[i][j] = image[oy][ox]; break;
      case CR_OUT: out_image[i][j] = color_out; break;
      case CR_ITE: out_image[i][j] = color_ite; break;
      case CR_ABS: out_image[i][j] = color_abs; break;
      }
    }
  }

  std::cerr << "\rAll done ! Saving to result.png" << std::endl;
  out_image.write("result.png");
  
  return 0;
}
