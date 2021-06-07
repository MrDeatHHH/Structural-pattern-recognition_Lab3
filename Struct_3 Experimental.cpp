#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono> 

using namespace cv;
using namespace std;
using namespace std::chrono;

// Dont touch these
const int modK = 5;
const int modNt = 4;
const double infinity = 10000000000;
const int colors_draw[modK + 1][3] = { {0, 0, 0}, {0, 255, 0}, {0, 255, 255}, {0, 0, 255}, {255, 0, 255}, {255, 0, 0} };

// Better dont touch these
const double color_scale = 1. / 100.;
const double weight = 0.5;

// Actual params, which can be changed
// Better start with epsilon around 10
// And give at least 500 first iterations
const int em_iter = 10;
const int first_iter = 2000;
const int iter = 200;
const double epsilon = 10;
// Iterations will stop when prev epsilon and cur epsilon wont differ more then on this value
const double accuracy = 0.01;

// Set hack_value to zero if u are not hacker
// Empirical rule: if current epsilon is close to zero
// That means that diffusion made all (or almost all) max arcs equal
// Which means that we can get solution from the updated weights of diffusion
// So instead of finding markup which epsilon close to best we can find best itself
const double hack_value = 0.;

const bool experimental = true;

// Percentages of image taken for calculating distribution
const double perc_h = 0.2;

// Aprior probs
const double aprior[3] = {9. / 20.,
                          1. / 10.,
	                      9. / 20.};

// Saving results
void save_and_show(int* res, const int width, const int height, string name, bool save = false)
{
	Mat* result = new Mat[3];
	for (int c = 0; c < 3; ++c)
	{
		result[c] = Mat::zeros(Size(width, height), CV_8UC1);
		for (int x = 0; x < width; ++x)
			for (int y = 0; y < height; ++y)
			{
				result[c].at<uchar>(y, x) = uchar(colors_draw[1 + res[x * height + y]][c]);
			}
	}

	Mat rez;
	vector<Mat> channels;

	channels.push_back(result[0]);
	channels.push_back(result[1]);
	channels.push_back(result[2]);

	merge(channels, rez);

	namedWindow(name, WINDOW_AUTOSIZE);
	cv::imshow(name, rez);
	if (save)
		imwrite(name + ".png", rez);

	delete[] result;
}

// Scalar mult
double vec_mult(double* x, double* y)
{
	double sum = 0.;
	for (int c = 0; c < 3; ++c)
		sum += x[c] * y[c];

	return sum;
}

// vec(x) * mat(a)
double* mat_mult(double* x, double** a)
{
	double* res = new double[3];
	for (int c = 0; c < 3; ++c)
		res[c] = vec_mult(x, a[c]);

	return res;
}

// q function
double q_func(int x[3], double* mus, double* eps, const int k)
{
	// TODO: check if this is correct
	double* x_ = new double[3];
	for (int c = 0; c < 3; ++c)
		x_[c] = x[c] * color_scale - mus[k * 3 + c];

	Mat mep(3, 3, CV_64FC1);
	for (int c = 0; c < 3; ++c)
		for (int c_ = 0; c_ < 3; ++c_)
			mep.at<double>(c, c_) = eps[k * 9 + c * 3 + c_];

	Mat inverse = mep.inv();

	double** ep = new double* [3];
	for (int c = 0; c < 3; ++c)
		ep[c] = new double[3];

	for (int c = 0; c < 3; ++c)
		for (int c_ = 0; c_ < 3; ++c_)
			ep[c][c_] = inverse.at<double>(c, c_);

	double* z = mat_mult(x_, ep);
	double result = vec_mult(x_, z);

	delete[] z;
	delete[] x_;

	for (int c = 0; c < 3; ++c)
		delete[] ep[c];
	delete[] ep;

	return -result;
}

double g_plus(const int x1, const int y1, const int x2, const int y2, const int k1, const int k2)
{
	if ((k1 == 0) && (k2 == 0))
	{
		return 0;
	}

	if ((k1 == 1) && (k2 == 1))
	{
		if (x1 != x2)
			return 0;
		else
			return -infinity;
	}

	if ((k1 == 2) && (k2 == 2))
	{
		return 0;
	}

	if ((k1 == 3) && (k2 == 3))
	{
		if (x1 != x2)
			return 0;
		else
			return -infinity;
	}

	if ((k1 == 4) && (k2 == 4))
	{
		return 0;
	}

	///////////////////////////////////////////////

	if ((k1 == 1) && (k2 == 0))
	{
		if (y1 > y2)
			return -infinity;
		else
			return -weight;
	}

	if ((k1 == 0) && (k2 == 1))
	{
		if (y2 > y1)
			return -infinity;
		else
			return -weight;
	}

	// -------------------------

	if ((k1 == 2) && (k2 == 0))
	{
		return -infinity;
	}

	if ((k1 == 0) && (k2 == 2))
	{
		return -infinity;
	}

	// --------------------------

	if ((k1 == 3) && (k2 == 0))
	{
		if (y1 > y2)
			return -infinity;
		else
			return -weight;
	}

	if ((k1 == 0) && (k2 == 3))
	{
		if (y2 > y1)
			return -infinity;
		else
			return -weight;
	}

	// ---------------------------

	if ((k1 == 4) && (k2 == 0))
	{
		return -infinity;
	}

	if ((k1 == 0) && (k2 == 4))
	{
		return -infinity;
	}

	// ----------------------------

	if ((k1 == 2) && (k2 == 1))
	{
		if (y1 > y2)
			return -infinity;
		else
			return -weight;
	}

	if ((k1 == 1) && (k2 == 2))
	{
		if (y2 > y1)
			return -infinity;
		else
			return -weight;
	}

	// ---------------------------
	
	if ((k1 == 3) && (k2 == 1))
	{
		if (y1 > y2)
			return -infinity;
		else
			return -weight;
	}

	if ((k1 == 1) && (k2 == 3))
	{
		if (y2 > y1)
			return -infinity;
		else
			return -weight;
	}

	// --------------------------

	if ((k1 == 4) && (k2 == 1))
	{
		return -infinity;
	}

	if ((k1 == 1) && (k2 == 4))
	{
		return -infinity;
	}

	// -------------------------

	if ((k1 == 3) && (k2 == 2))
	{
		if (y1 > y2)
			return -infinity;
		else
			return -weight;
	}

	if ((k1 == 2) && (k2 == 3))
	{
		if (y2 > y1)
			return -infinity;
		else
			return -weight;
	}

	// -------------------------

	if ((k1 == 4) && (k2 == 2))
	{
		return -infinity;
	}

	if ((k1 == 2) && (k2 == 4))
	{
		return -infinity;
	}

	// -------------------------

	if ((k1 == 4) && (k2 == 3))
	{
		if (y1 > y2)
			return -infinity;
		else
			return -weight;
	}

	if ((k1 == 3) && (k2 == 4))
	{
		if (y2 > y1)
			return -infinity;
		else
			return -weight;
	}

	// -------------------------

	cout << "Something is wrong... I can feel it" << endl;
	return -infinity;
}

double* calculate_mu(const int height, const int width, const int col, int* colors, int* colors_mask)
{
	int amount = 0;

	// Initialize result
	double* result = new double[3]();

	// To be safe
	for (int c = 0; c < 3; ++c)
		result[c] = 0.;

	// Sum all the pixel colors
	for (int t = 0; t < width * height; ++t)
	{
		if (colors_mask[t * 3 + col] == 255)
		{
			for (int c = 0; c < 3; ++c)
				result[c] += double(colors[t * 3 + c]) * color_scale;
			amount++;
		}
	}

	// Devide by amount
	for (int c = 0; c < 3; ++c)
		result[c] /= double(amount);

	return result;
}

double* calculate_ep(const int height, const int width, const int col, int* colors, int* colors_mask, double* mu)
{
	int amount = 0;

	// Initialize result
	double* result = new double[3 * 3]();

	// To be safe
	for (int c1 = 0; c1 < 3; ++c1)
		for (int c2 = 0; c2 < 3; ++c2)
			result[c1 * 3 + c2] = 0.;

	// Sum all the elements
	for (int t = 0; t < width * height; ++t)
	{
		if (colors_mask[t * 3 + col] == 255)
		{
			double x_mu[3];
			for (int c = 0; c < 3; ++c)
			{
				x_mu[c] = colors[t * 3 + c] * color_scale - mu[c];
			}

			for (int c1 = 0; c1 < 3; ++c1)
				for (int c2 = 0; c2 < 3; ++c2)
				{
					// TODO: check if this is correct
					result[c1 * 3 + c2] += x_mu[c1] * x_mu[c2];
				}
			amount++;
		}
	}
	
	// Devide by amount
	for (int c1 = 0; c1 < 3; ++c1)
		for (int c2 = 0; c2 < 3; ++c2)
			result[c1 * 3 + c2] /= amount;

	return result;
}

void get_distributions(double* mus, double* eps,
	const int height, const int width, int* colors, int const classes)
{
	int* colors_mask = new int[height * width * 3]();

	for (int x = 0; x < width; ++x)
	{
		for (int y = 0; y < height * perc_h; ++y)
			colors_mask[(x * height + y) * 3 + 0] = 255;

		for (int y = int((1 - perc_h) * height); y < height; ++y)
			colors_mask[(x * height + y) * 3 + 1] = 255;
	}

	// Calculate all mus
	double* mu_0 = calculate_mu(height, width, 1, colors, colors_mask);
	double* mu_2 = calculate_mu(height, width, 0, colors, colors_mask);
	double* mu_1 = new double[3];
	
	for (int d = 0; d < 3; ++d)
	{
		if (abs(mu_0[d] - mu_2[d]) > 128 * color_scale)
		{
			mu_1[d] = (mu_0[d] + mu_2[d]) / 2.;
			mu_1[d] = (mu_0[d] + mu_2[d]) / 2.;
		}
		else
		{
			//double ar[2] = { 0., 255. * color_scale };
			double ar[1] = { 0. };
			int ind = -1;
			double max = -1;

			for (int d_ = 0; d_ < 1; ++d_)
			{
				double a = pow(abs(ar[d_] - mu_0[d]) / color_scale, 0.5);
				double b = pow(abs(ar[d_] - mu_2[d]) / color_scale, 0.5);
				double cur = a + b;

				if (cur > max)
				{
					ind = d_;
					max = cur;
				}
			}

			mu_1[d] = ar[ind];
		}
	}

	for (int i = 0; i < 3; ++i)
	{
		mus[0 + i] = mu_0[i];
		mus[3 + i] = mu_1[i];
		mus[6 + i] = mu_2[i];
	}

	// Calculate all eps
	double* ep_0 = calculate_ep(height, width, 1, colors, colors_mask, mu_0);
	double* ep_2 = calculate_ep(height, width, 0, colors, colors_mask, mu_2);

	for (int i = 0; i < 9; ++i)
	{
		eps[0 + i] = ep_0[i];
		if (i % 3 == int(i / 3))
			eps[9 + i] = (ep_0[i] + ep_2[i]);
		else
			eps[9 + i] = 0.;
		eps[18 + i] = ep_2[i];
	}

	delete[] mu_0;
	delete[] mu_1;
	delete[] mu_2;
	delete[] ep_0;
	delete[] ep_2;
}

void EM(int* colors, const int height, const int width, double* mus, double* eps)
{
	const int classes = 3;
	const int modT = width * height;

	// Initialize probs
	double* p_k_x = new double[classes * modT]();
	double* p_k = new double[classes]();
	for (int c = 0; c < 3; ++c)
		p_k[c] = aprior[c];
	double* p_x_k = new double[modT * classes]();

	// For saving and further processing
	double* mus_ = new double[classes * 3]();
	double* eps_ = new double[classes * 9]();
	get_distributions(mus_, eps_, height, width, colors, classes);

	double ref[6];
	for (int d = 0; d < 2; ++d)
		for (int d_ = 0; d_ < 3; ++d_)
			ref[d * 3 + d_] = mus_[d * 6 + d_];
	
	cout << " - - - - - - - - - - - - - - - " << endl;

	for (int i = 0; i < 3; ++i)
		cout << mus_[i] << " " << endl;
	cout << endl;

	for (int i = 0; i < 3; ++i)
		cout << mus_[3 + i] << " " << endl;
	cout << endl;

	for (int i = 0; i < 3; ++i)
		cout << mus_[6 + i] << " " << endl;
	cout << endl;

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
			cout << eps_[i * 3 + j] << " ";
		cout << endl;
	}
	cout << endl;

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
			cout << eps_[9 + i * 3 + j] << " ";
		cout << endl;
	}
	cout << endl;

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
			cout << eps_[18 + i * 3 + j] << " ";
		cout << endl;
	}
	cout << endl;
	
	for (int it = 0; it < em_iter; ++it)
	{
		cout << "EM --- " << it << " / " << em_iter << endl;

		// First step
		for (int t = 0; t < modT; ++t)
		{
			double sum = 0;
			for (int c = 0; c < classes; ++c)
			{
				int x[3] = { colors[t * 3], colors[t * 3 + 1], colors[t * 3 + 2] };
				if (q_func(x, mus_, eps_, c) > 50)
					p_x_k[t * classes + c] = infinity;
				else
					p_x_k[t * classes + c] = exp(0.5 * q_func(x, mus_, eps_, c));
				sum += p_x_k[t * classes + c];
			}

			for (int c = 0; c < classes; ++c)
				p_x_k[t * classes + c] /= sum;
		}

		// Second step
		for (int c = 0; c < classes; ++c)
		{
			for (int t = 0; t < modT; ++t)
			{
				double sum = 0;
				for (int c_ = 0; c_ < classes; ++c_)
					sum += p_x_k[t * classes + c_] * p_k[c_];

				p_k_x[c * modT + t] = p_x_k[t * classes + c] * p_k[c] / sum;
			}
		}

		// Third step
		for (int c = 0; c < classes; ++c)
		{
			double sum = 0;
			for (int t = 0; t < modT; ++t)
				sum += p_k_x[c * modT + t];
			p_k[c] = sum / double(modT);
		}

		// Claculate mus_
		for (int c = 0; c < classes; ++c)
		{
			for (int d = 0; d < 3; ++d)
			{
				mus_[c * 3 + d] = 0;
				double sum = 0;

				for (int t = 0; t < modT; ++t)
				{
					mus_[c * 3 + d] += p_k_x[c * modT + t] * colors[t * 3 + d];
					sum += p_k_x[c * modT + t];
				}

				mus_[c * 3 + d] *= color_scale;
				mus_[c * 3 + d] /= sum;
			}
		}

		// Claculate eps_
		for (int c = 0; c < classes; ++c)
		{
			for (int d = 0; d < 3; ++d)
			{
				for (int d_ = 0; d_ < 3; ++d_)
				{
					eps_[c * 9 + d * 3 + d_] = 0;
					double sum = 0;
					for (int t = 0; t < modT; ++t)
					{
						eps_[c * 9 + d * 3 + d_] += p_k_x[c * modT + t] * double(colors[t * 3 + d] * color_scale - mus_[c * 3 + d]) * double(colors[t * 3 + d_] * color_scale - mus_[c * 3 + d_]);
						sum += p_k_x[c * modT + t];
					}

					eps_[c * 9 + d * 3 + d_] /= sum;
				}
			}
		}

		
		// Experimental
		if (experimental)
		{
			for (int c = 0; c < classes; ++c)
			{
				for (int d = 0; d < 3; ++d)
					for (int d_ = 0; d_ < 3; ++d_)
						eps_[c * 9 + d * 3 + d_] = (eps_[((c + 1) % classes) * 9 + d * 3 + d_] + eps_[((c + 2) % classes) * 9 + d * 3 + d_]) * 0.5;
			}
		}
	}

	int di[3] = { -1, -1, -1 };

	for (int d = 0; d < 2; ++d)
	{
		double min_dif = 1000000;
		int ind = -1;
		for (int d_ = 0; d_ < 3; ++d_)
		{
			if (d_ != di[0])
			{
				double cur = 0.;
				for (int c = 0; c < 3; ++c)
				{
					cur += pow(abs(mus_[d_ * 3 + c] - ref[d * 3 + c]) / color_scale, 0.5);
				}

				if (cur < min_dif)
				{
					ind = d_;
					min_dif = cur;
				}
			}
		}

		di[d * 2] = ind;
	}
	di[1] = 3 - di[0] - di[2];
	for (int i = 0; i < 2; ++i)
		cout << di[i] << " --- ";
	cout << di[2] << endl;

	for (int d = 0; d < 3; ++d)
	{
		mus[0 * 3 + d] = mus_[di[0] * 3 + d];
		mus[1 * 3 + d] = mus_[di[1] * 3 + d];
		mus[2 * 3 + d] = mus_[di[1] * 3 + d];
		mus[3 * 3 + d] = mus_[di[2] * 3 + d];
		mus[4 * 3 + d] = mus_[di[2] * 3 + d];

		for (int d_ = 0; d_ < 3; ++d_)
		{
			eps[0 * 9 + d * 3 + d_] = eps_[di[0] * 9 + d * 3 + d_];
			eps[1 * 9 + d * 3 + d_] = eps_[di[1] * 9 + d * 3 + d_];
			eps[2 * 9 + d * 3 + d_] = eps_[di[1] * 9 + d * 3 + d_];
			eps[3 * 9 + d * 3 + d_] = eps_[di[2] * 9 + d * 3 + d_];
			eps[4 * 9 + d * 3 + d_] = eps_[di[2] * 9 + d * 3 + d_];
		}
	}

	for (int c = 0; c < 3; ++c)
		cout << " -------------------- " << p_k[c] << endl;

	delete[] p_k_x;
	delete[] p_k;
	delete[] p_x_k;
	delete[] mus_;
	delete[] eps_;
}

int amount_of_neightbors_plus(bool left, bool right, bool top, bool bottom)
{
	int result = 0;

	// Checking left side
	if (left)
		result++;

	// Checking right side
	if (right)
		result++;

	// Checking top pixel
	if (top)
		result++;

	// Checking bottom pixel
	if (bottom)
		result++;

	return result;
}

void get_neighbors_plus(int& nt, int* tau, int x, int y, int w, int h)
{
	bool left = (x != 0);
	bool right = (x != w - 1);
	bool top = (y != 0);
	bool bottom = (y != h - 1);

	// Calculate amount of neightbors
	nt = amount_of_neightbors_plus(left, right, top, bottom);

	int Index = 0;

	// Checking left side
	if (left)
	{
		tau[x * h * modNt + y * modNt + Index] = (x - 1) * h + y;
		Index++;
	}

	// Checking right side
	if (right)
	{
		tau[x * h * modNt + y * modNt + Index] = (x + 1) * h + y;
		Index++;
	}

	// Checking top pixel
	if (top)
	{
		tau[x * h * modNt + y * modNt + Index] = x * h + (y - 1);
		Index++;
	}

	// Checking bottom pixel
	if (bottom)
	{
		tau[x * h * modNt + y * modNt + Index] = x * h + (y + 1);
	}
}

int find(int* arr, const int start, const int length, const int t)
{
	for (int i = 0; i < length; ++i)
		if (arr[start + i] == t)
			return i;

	cout << "Something is wrong... I can feel it" << endl;
	return -1;
}

inline int get_g_ind(const int t, const int t__, const int height)
{
	return 2 * (abs(t - t__) == height) + (t__ > t);
}

inline int get_q_ind(const int t, int* colors)
{
	return ((colors[t * 3] * 256 + colors[t * 3 + 1]) * 256 + colors[t * 3 + 2]) * modK;
}

void diffusion(const int iter, double* mus, double* eps,
	int* nt, int* tau,
	double* phi, const int width, const int height,
	int* colors, double* g, double* q)
{
	const int modT = width * height;
	for (int it = 0; it < iter; ++it)
	{
		cout << "Diffusion --- " << it << " / " << iter << endl;
		for (int t = 0; t < modT; ++t)
		{
			const int _t = t * modNt * modK;
			for (int c = 0; c < modK; ++c)
			{
				// Finding k_best for each neighbor
				int* k_star = new int[nt[t]];
				for (int t_ = 0; t_ < nt[t]; ++t_)
				{
					int t__ = tau[t * modNt + t_];
					int k_best = -1;
					double sum_best = -infinity * 100.;
					for (int c_ = 0; c_ < modK; ++c_)
					{
						double sum = g[get_g_ind(t, t__, height) * modK * modK + c * modK + c_] -
							phi[_t + t_ * modK + c] -
							phi[t__ * modNt * modK + find(tau, t__ * modNt, nt[t__], t) * modK + c_];
						if (sum > sum_best)
						{
							k_best = c_;
							sum_best = sum;
						}
					}

					k_star[t_] = k_best;
				}
				// Calculating Constant C for further update of phi
				double Con = 0.;

				for (int t_ = 0; t_ < nt[t]; ++t_)
				{
					int t__ = tau[t * modNt + t_];
					Con += g[get_g_ind(t, t__, height) * modK * modK + c * modK + k_star[t_]] -
						phi[t__ * modNt * modK + find(tau, t__ * modNt, nt[t__], t) * modK + k_star[t_]];
				}

				Con += q[get_q_ind(t, colors) + c];
				Con /= double(nt[t]);

				// Updating phi
				for (int t_ = 0; t_ < nt[t]; ++t_)
				{
					int t__ = tau[t * modNt + t_];
					phi[_t + t_ * modK + c] = g[get_g_ind(t, t__, height) * modK * modK + c * modK + k_star[t_]] -
						phi[t__ * modNt * modK + find(tau, t__ * modNt, nt[t__], t) * modK + k_star[t_]] - Con;
				}

				delete[] k_star;
			}
		}
	}
}

int* get_result(int* nt, int* tau, double* phi, const int width, const int height, double* g)
{
	const int modT = width * height;
	int* result = new int[modT];

	for (int t = 0; t < modT; ++t)
	{
		const int _t = t * modNt * modK;
		// Finding k_best for first neightbor
		int t__ = tau[t * modNt];
		int k_best = -1;
		double sum_best = -infinity * 100;
		for (int c = 0; c < modK; ++c)
		{
			int k__best = -1;
			double sum__best = -infinity * 100;
			for (int c_ = 0; c_ < modK; ++c_)
			{
				double sum = g[get_g_ind(t, t__, height) * modK * modK + c * modK + c_] -
					phi[_t + c] -
					phi[t__ * modNt * modK + find(tau, t__ * modNt, nt[t__], t) * modK + c_];
				if (sum > sum__best)
				{
					k__best = c_;
					sum__best = sum;
				}
			}
			if (sum__best > sum_best)
			{
				k_best = c;
				sum_best = sum__best;
			}
		}

		result[t] = k_best;
	}

	return result;
}

void get_or_and_problem(bool*& gs, bool*& qs, int* nt, int* tau, double* phi, const int width, const int height, double* mus, double* eps, int* colors, double epsilon, double* g, double* q)
{
	// Inititalize needed variable
	const int modT = width * height;

	// Initialize qs and gs
	qs = new bool[modT * modK];
	gs = new bool[modT * modNt * modK * modK];

	// TODO: check if we should compare with gs, not other qs
	// Calculating qs
	for (int t = 0; t < modT; ++t)
	{
		const int _t = t * modNt * modK;
		// Finding max qs[t][k*] and leaving only those qs[t][k], for which holds (qs[t][k*] - qs[t][k] < epsilon)
		double max = -infinity;
		double* current_q = new double[modK];
		for (int c = 0; c < modK; ++c)
		{
			// Calculating current q
			current_q[c] = q[get_q_ind(t, colors) + c];
			for (int t_ = 0; t_ < nt[t]; ++t_)
				current_q[c] += phi[_t + t_ * modK + c];

			// Comparing to max
			if (current_q[c] > max)
				max = current_q[c];
		}

		// Calculating qs[t][k]
		for (int c = 0; c < modK; ++c)
			qs[t * modK + c] = (abs(max - current_q[c]) < epsilon);
		
		// Delete current qs
		delete[] current_q;
	}

	// Calculating gs
	for (int t = 0; t < modT; ++t)
	{
		const int _t = t * modNt * modK;
		// Find max gs[t][t_*][k*][k_*] and leaving only those gs[t][t_][k][k_], for which holds (gs[t][t_*][k*][k_*] - gs[t][t_][k][k_] < epsilon)
		double max = -infinity;
		double** current_g = new double* [nt[t]];

		for (int t_ = 0; t_ < nt[t]; ++t_)
		{
			int t__ = tau[t * modNt + t_];
			current_g[t_] = new double[modK * modK];
			for (int c = 0; c < modK; ++c)
			{
				for (int c_ = 0; c_ < modK; ++c_)
				{
					// Calculating current g
					current_g[t_][c * modK + c_] = g[get_g_ind(t, t__, height) * modK * modK + c * modK + c_] -
						phi[_t + t_ * modK + c] -
						phi[t__ * modNt * modK + find(tau, t__ * modNt, nt[t__], t) * modK + c_];

					// Comparing to max
					if (current_g[t_][c * modK + c_] > max)
						max = current_g[t_][c * modK + c_];
				}
			}
		}

		// Calculating gs[t][t_][k][k_]
		for (int t_ = 0; t_ < nt[t]; ++t_)
			for (int c = 0; c < modK; ++c)
				for (int c_ = 0; c_ < modK; ++c_)
					gs[_t * modK + t_ * modK * modK + c * modK + c_] = (abs(max - current_g[t_][c * modK + c_]) < epsilon);

		// Delete current gs
		for (int t_ = 0; t_ < nt[t]; ++t_)
			delete[] current_g[t_];
		delete[] current_g;
	}

}

void cross(bool* gs, bool* qs, int* nt, int* tau, const int width, const int height)
{
	// Inititalize needed variables
	const int modT = width * height;
	bool changed = true;

	const int modK2 = modK * modK;
	// Repeat untill something changes
	while (changed)
	{
		changed = false;
		// Update qs
		for (int t = 0; t < modT; ++t)
		{
			const int _t = t * modNt * modK2;
			for (int c = 0; c < modK; ++c)
			{
				if (qs[t * modK + c])
					for (int t_ = 0; t_ < nt[t]; ++t_)
					{
						// Calculate or over all possible ks
						int t__ = tau[t * modNt + t_];
						bool result = false;
						for (int c_ = 0; c_ < modK; ++c_)
							result = result || (gs[_t + t_ * modK2 + c * modK + c_] && qs[t__ * modK + c_]);

						// If the result is false, then whole AND will be false
						if (!result)
						{
							qs[t * modK + c] = false;
							changed = true;
							break;
						}
					}
			}
		}

		// Update gs
		for (int t = 0; t < modT; ++t)
		{
			const int _t = t * modNt * modK2;
			for (int t_ = 0; t_ < nt[t]; ++t_)
			{
				int t__ = tau[t * modNt + t_];
				for (int c = 0; c < modK; ++c)
				{
					for (int c_ = 0; c_ < modK; ++c_)
						if (gs[_t + t_ * modK2 + c * modK + c_])
						{
							gs[_t + t_ * modK2 + c * modK + c_] = qs[t * modK + c] && qs[t__ * modK + c_];
							if (!gs[_t + t_ * modK2 + c * modK + c_])
								changed = true;
						}
				}
			}
		}
	}
}

bool f(bool* qs)
{
	// Checks if there is a markup after crossing
	bool check = false;
	for (int c = 0; c < modK; ++c)
		check = check || qs[c];
	return check;
}

bool self_control(int* answer, bool* gs, bool* qs, int* nt, int* tau, const int width, const int height)
{
	if (f(qs))
	{
		bool result = false;

		// Inititalize needed variables
		const int modT = width * height;

		for (int t = 0; t < modT; ++t)
			answer[t] = -1;

		const int size_qs = modT * modK;
		const int size_gs = modT * modNt * modK * modK;

		// Initialize qs_ and gs_
		bool* qs_ = new bool[size_qs];
		bool* gs_ = new bool[size_gs];

		double counter_copy = 0;
		double counter_cross = 0;
		double counter_trash = 0;

		// Start main loop
		for (int t = 0; t < modT; ++t)
		{
			if (t % 100 == 0)
				cout << "Self control --- " << t << " / " << modT << endl;
			result = false;
			for (int c = 0; c < modK; ++c)
			{
				// If qs[t][k] is true then check if there is a markup after cross
				if (qs[t * modK + c])
				{
					auto mark = high_resolution_clock::now();
					std::copy(qs, qs + size_qs, qs_);
					std::copy(gs, gs + size_gs, gs_);
					counter_copy += (double(duration_cast<microseconds>(high_resolution_clock::now() - mark).count()) / 1000000.);

					mark = high_resolution_clock::now();
					// Making other qs equal to false
					for (int c_ = 0; c_ < modK; ++c_)
						if (c != c_)
							qs_[t * modK + c_] = false;
					counter_trash += (double(duration_cast<microseconds>(high_resolution_clock::now() - mark).count()) / 1000000.);

					mark = high_resolution_clock::now();
					// Using cross
					cross(gs_, qs_, nt, tau, width, height);
					counter_cross += (double(duration_cast<microseconds>(high_resolution_clock::now() - mark).count()) / 1000000.);

					// Checking if there is markup
					if (f(qs_))
					{
						mark = high_resolution_clock::now();
						result = true;
						answer[t] = c;
						for (int c_ = 0; c_ < modK; ++c_)
							qs[t * modK + c_] = (c == c_);
						counter_trash += (double(duration_cast<microseconds>(high_resolution_clock::now() - mark).count()) / 1000000.);
						break;
					}

				}
			}

			// If we didnt find markup then break
			if (!result)
				break;
		}

		// Delete qs_ and gs_
		delete[] qs_;
		delete[] gs_;

		double sum_time = counter_trash + counter_cross + counter_copy;
		cout << "Time used for trash: " << counter_trash / sum_time << " %" << endl;
		cout << "Time used for cross: " << counter_cross / sum_time << " %" << endl;
		cout << "Time used for copy: " << counter_copy / sum_time << " %" << endl;
		cout << "Time used for self control: " << sum_time << endl;

		return result;
	}
	else
	{
		return false;
	}
}

int* iterations(const int first_iter, const int iter,
	const double accuracy, double epsilon,
	double* mus, double* eps,
	int* nt, int* tau,
	const int width, const int height,
	int* colors,
	double* g, double* q)
{
	double prev_epsilon = 2 * epsilon;
	double current_epsilon = epsilon;
	const int modT = width * height;
	bool stop = false;
	int counter = 0;
	int* last_res = new int[modT]();
	for (int t = 0; t < modT; ++t)
		last_res[t] = -1;
	int* current_res = new int[modT]();
	for (int t = 0; t < modT; ++t)
		current_res[t] = -1;

	// Initialize phi
	double* phi = new double[modT * modNt * modK]();

	// First Diffusion
	diffusion(first_iter, mus, eps, nt, tau, phi, width, height, colors, g, q);

	while (!stop)
	{
		cout << prev_epsilon << " - " << current_epsilon << " <> " << accuracy << endl;
		// Get or and problem
		bool* gs;
		bool* qs;
		get_or_and_problem(gs, qs, nt, tau, phi, width, height, mus, eps, colors, current_epsilon, g, q);

		// Use crossing
		cross(gs, qs, nt, tau, width, height);

		bool check;
		bool have_result = false;

		if (current_epsilon < 0.01 && hack_value == 0.)
			cout << "Consider using hacks" << endl;

		if ((prev_epsilon - current_epsilon) < 2 * accuracy)
		{
			check = f(qs);
			if (check)
			{
				if ((prev_epsilon - current_epsilon) > hack_value)
					check = self_control(current_res, gs, qs, nt, tau, width, height);
				else
				{
					check = f(qs);
					current_res = get_result(nt, tau, phi, width, height, g);
				}
				have_result = true;
			}
		}
		else
		{
			check = f(qs);
		}

		if (check)
		{
			if (have_result)
				for (int t = 0; t < modT; ++t)
					last_res[t] = current_res[t];

			prev_epsilon = current_epsilon;
			current_epsilon *= 0.5;
		}
		else
		{
			current_epsilon = (prev_epsilon + current_epsilon) * 0.5;
		}

		if ((prev_epsilon - current_epsilon) < accuracy)
		{
			stop = true;
		}
		else
		{
			diffusion(iter, mus, eps, nt, tau, phi, width, height, colors, g, q);
		}

		delete[] qs;
		delete[] gs;
	}

	if (last_res[0] == -1)
	{
		/*
		cout << "Returning previous result" << endl;
		bool ok = true;
		for (int i = 0; i < modT; ++i)
			ok = ok && (current_res[i] != -1);

		if (ok)
			for (int i = 0; i < modT; ++i)
				last_res[i] = current_res[i];
		else
		{
			cout << "Not enought diffusion iterations" << endl;
			cout << "Didnt find solution with selfcontrol, using diffusion weights to get possible result" << endl;
			last_res = get_result(nt, tau, phi, width, height, g);
		}
		*/
		cout << "Not enought diffusion iterations" << endl;
		cout << "Didnt find solution with selfcontrol, using diffusion weights to get possible result" << endl;
		last_res = get_result(nt, tau, phi, width, height, g);
	}

	delete[] current_res;

	return last_res;
}

int main()
{
	Mat image_, image[4];
	image_ = imread("1.jpg", IMREAD_UNCHANGED);
	split(image_, image);

	auto start = high_resolution_clock::now();

	const int height = image[0].size().height;
	const int width = image[0].size().width;

	// Get array from Mat
	int* colors = new int[width * height * 3];
	for (int x = 0; x < width; ++x)
	{
		const int x_ = x * height * 3;
		for (int y = 0; y < height; ++y)
		{
			const int y_ = y * 3;
			for (int c = 0; c < 3; ++c)
				colors[x_ + y_ + c] = int(image[c].at<uchar>(y, x));
		}
	}

	// Form single array for mus and eps
	// 0 - down
	// 1 - border_mid
	// 2 - mid
	// 3 - border_up
	// 4 - up
	double* mus = new double[modK * 3];
	double* eps = new double[modK * 3 * 3];

	EM(colors, height, width, mus, eps);
	
	for (int i = 0; i < 3; ++i)
		cout << mus[i] << " " << endl;
	cout << endl;

	for (int i = 0; i < 3; ++i)
		cout << mus[6 + i] << " " << endl;
	cout << endl;

	for (int i = 0; i < 3; ++i)
		cout << mus[12 + i] << " " << endl;
	cout << endl;

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
			cout << eps[i * 3 + j] << " ";
		cout << endl;
	}
	cout << endl;

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
			cout << eps[18 + i * 3 + j] << " ";
		cout << endl;
	}
	cout << endl;

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
			cout << eps[36 + i * 3 + j] << " ";
		cout << endl;
	}
	cout << endl;

	if (true)
	{
		auto mark = high_resolution_clock::now();
		cout << "Precalculating g" << endl;
		// Initialize q and g
		double* g = new double[modNt * modK * modK];

		for (int c = 0; c < modK; ++c)
			for (int c_ = 0; c_ < modK; ++c_)
				g[0 * modK * modK + c * modK + c_] = g_plus(1, 1, 1, 0, c, c_);

		for (int c = 0; c < modK; ++c)
			for (int c_ = 0; c_ < modK; ++c_)
				g[1 * modK * modK + c * modK + c_] = g_plus(1, 1, 1, 2, c, c_);

		for (int c = 0; c < modK; ++c)
			for (int c_ = 0; c_ < modK; ++c_)
				g[2 * modK * modK + c * modK + c_] = g_plus(1, 1, 0, 1, c, c_);

		for (int c = 0; c < modK; ++c)
			for (int c_ = 0; c_ < modK; ++c_)
				g[3 * modK * modK + c * modK + c_] = g_plus(1, 1, 2, 1, c, c_);
		cout << "Done" << endl;
		cout << "Time used: " << double(duration_cast<microseconds>(high_resolution_clock::now() - mark).count()) / 1000000. << endl;

		mark = high_resolution_clock::now();
		cout << "Precalculating q" << endl;
		double* q = new double[256 * 256 * 256 * modK]();
		for (int t = 0; t < height * width; ++t)
		{
			int x[3] = { colors[t * 3], colors[t * 3 + 1], colors[t * 3 + 2] };
			const int ind = x[0] * 256 * 256 * modK + x[1] * 256 * modK + x[2] * modK;
			for (int c = 0; c < modK; ++c)
				q[ind + c] = q_func(x, mus, eps, c);
		}
		cout << "Done" << endl;
		cout << "Time used: " << double(duration_cast<microseconds>(high_resolution_clock::now() - mark).count()) / 1000000. << endl;

		// Create neighbour structure
		const int modT = width * height;
		int* tau = new int[modT * modNt];
		int* nt = new int[modT];
		for (int x = 0; x < width; ++x)
		{
			for (int y = 0; y < height; ++y)
				get_neighbors_plus(nt[x * height + y], tau, x, y, width, height);
		}

		int* res = iterations(first_iter, iter, accuracy, epsilon, mus, eps, nt, tau, width, height, colors, g, q);

		// Measuring time taken
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		cout << "Time used: " << double(duration.count()) / 1000000. << endl;

		if (res[0] != -1)
		{
			save_and_show(res, width, height, "res", true);
		}
		else
		{
			cout << "Bad params" << endl;
			cout << "Try to change epsilon and/or accuracy" << endl;
		}
	}

	waitKey(0);
	return 0;
}