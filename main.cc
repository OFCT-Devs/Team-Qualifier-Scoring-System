#include "ceres/ceres.h"
#include "glog/logging.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>

using ceres::AutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

// F(x) = a log(x - b) - c log(-x + d) + f ~= y_

constexpr size_t mappool_size = 10;

static size_t pivot = 0;

struct User {
  std::string username;
  size_t usernum;
  std::array<double, mappool_size> data;

  User() = default;
  User(std::string username, size_t usernum, std::array<size_t, mappool_size> data) : username(std::move(username)), usernum(usernum) {
    for(int i = 0; i < mappool_size; i++) {
      this->data[i] = data[i] / 100000.0;
    }
  }
  double &operator[](size_t idx) { return data[idx]; }
  bool operator<(User const &usr) { return data[pivot] < usr.data[pivot]; }
  bool operator>(User const &usr) { return data[pivot] > usr.data[pivot]; }
};

std::vector<User> user;

constexpr int kNumObservations = 54;

void input() {
  std::ifstream in("input.csv");
  std::string str;
  // input format:
  // name1,score,score,...
  // ..

  while(!in.eof()) {
    std::getline(in, str);
    auto const l = str.size();
    std::string username;
    size_t usernum;
    std::array<size_t, mappool_size> data;
    size_t cnt = 0;
    size_t prev = 0;
    for(size_t i = 0; i < l; i++) {
      if(cnt == mappool_size + 1) {
        data[9] = std::stoi(str.substr(prev));
        break;
      }
      if(str[i] == ',') {
        if(cnt == 0) {
          username = str.substr(0, i);
          prev = i + 1;
          cnt = 1;
        }
        else if(cnt == 1) {
          usernum = std::stoi(str.substr(prev, i - prev - 1));
          prev = i + 1;
          cnt = 2;
        }
        else {
          data[cnt - 2] = std::stoi(str.substr(prev, i - prev - 1));
          prev = i + 1;
          cnt++;
        }
      }
    }
    user.emplace_back(std::move(username), usernum, std::move(data));
  }
  in.close();
}

auto const data_max_f() {
  auto max = user[0][pivot];
  for(int i = 1; i < kNumObservations; i++) {
    max = max > user[i][pivot] ? max : user[i][pivot];
  }
  return max;
}
auto const data_min_f() {
  auto min = user[0][pivot];
  for(int i = 1; i < kNumObservations; i++) {
    min = min < user[i][pivot] ? min : user[i][pivot];
  }
  return min;
}

double sn; // = data_max_f();
double s1; // = data_min_f();

struct Residual {
  Residual(double x, double y) : x_(x), y_(y) {}
  template<typename T> using pointer = T const *const;
  template <typename T>
  bool operator()(pointer<T> b, pointer<T> c, pointer<T> d, T* residual) const {
    auto const a = ((sn - s1 - c[0] * log((d[0] - 1.0) / (d[0] - (double)kNumObservations))) / log(((double)kNumObservations - b[0]) / (1.0 - b[0])));
    auto const f = s1 - a * log(1.0 - b[0]) + c[0] * log(-1.0 + d[0]);

    residual[0] = y_ - (a * log(x_ - b[0]) - c[0] * log(-x_ + d[0]) + f);
    bool status = a > 0.0 &&
                  c[0] > 0.0 &&
                  b[0] < x_ &&
                  d[0] > x_;
    return status;
  }
 private:
  const double x_;
  const double y_;
};

struct equation {
  double a, b, c, d, f;
  equation(double a, double b, double c, double d, double f) : a(a), b(b), c(c), d(d), f(f) {}
};
std::map<std::string, std::pair<size_t, std::array<double, mappool_size>>> map;
std::vector<equation> eqn;

int main(int argc, char** argv) {

  input();

  for(auto& now: user) {
    std::cout << now.username << ", " << now.usernum << ": ";
    for(int i = 0; i < mappool_size; i++) {
      std::cout << now.data[i] << ' ';
    }
    std::cout << '\n';
  }

  std::ofstream ofs("output.csv");

  google::InitGoogleLogging(argv[0]);
  
  for(size_t i = 0; i < kNumObservations; i++) {
    map[user[i].username] = std::make_pair(user[i].usernum, std::array<double, mappool_size>());
  }
  for(pivot = 0; pivot < 10; pivot++) {
    double b = 0.0;
    double c = 1e-6;
    double d = kNumObservations + 1.0;
    Problem problem;
    sn = data_max_f();
    s1 = data_min_f();
    std::cout << "sn, s1 = " << sn << ", " << s1 << std::endl;
    std::sort(user.begin(), user.end());

    for (int i = 0; i < kNumObservations; ++i) {
      CostFunction* cost_function =
          new AutoDiffCostFunction<Residual, 1, 1, 1, 1>(
              new Residual(i + 1, user[i][pivot]));
      problem.AddResidualBlock(cost_function, new CauchyLoss(0.5), &b, &c, &d);
    }
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100000;
    options.min_trust_region_radius = 1e-12;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    double const a = (sn - s1 - c * log((d - 1) / (d - kNumObservations))) / log((kNumObservations - b) / (1 - b));
    double const f = s1 - a * log(1 - b) + c * log(-1 + d);

    std::cout << a << ' ' << b << ' ' << c << ' ' << d << ' ' << f << std::endl;
    eqn.emplace_back(a, b, c, d, f);

    auto const func = [=](double x) { return a * log(x - b) - c * log(d - x) + f; };

    double result[kNumObservations] = {};
    result[0] = 1.0;
    result[kNumObservations - 1] = kNumObservations;

    for(int i = 1; i < kNumObservations - 1; i++) {
      double l = 1.0, r = kNumObservations;
      while(r - l > 1e-6) {
        auto const m = (l + r) / 2;
        auto const res = func(m);
        if(res == user[i][pivot]) {
          break;
        }
        if(res < user[i][pivot]) {
          l = m;
        }
        else {
          r = m;
        }
      }
      result[i] = (l + r) / 2;
    }

    std::cout << std::endl;

    for(int i = 0; i < kNumObservations; i++) {
      std::cout << user[i].username << ": " << kNumObservations + 1 - result[i] << std::endl;
      map[user[i].username].second[pivot] = kNumObservations + 1 - result[i];
    }
  }

  for(auto const &now: map) {
    ofs << now.first << ',' << now.second.first;
    for(size_t i = 0; i < mappool_size; i++) {
      ofs << ',' << now.second.second[i];
    }
    ofs << std::endl;
  }
  for(auto const &now: eqn) {
    ofs << now.a << ',' << now.b << ',' << now.c << ',' << now.d << ',' << now.f << std::endl;
  }

  ofs.close();
  return 0;
}