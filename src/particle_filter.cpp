/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	// set number of particle
	num_particles = 100;

	// set size of weights and paticles
	weights.resize(num_particles);
	particles.resize(num_particles);

	// create normal distributions (with mean = x, y, theta)
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// initialize all particles' configurations with uncertainty due to GPS as well as their weights
	for(unsigned int i=0; i<num_particles; ++i){

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = 1.0;

		weights[i] = particles[i].weight;

	}

	// initialization is done now
	is_initialized = true;

}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	// for generation noise
	default_random_engine gen;

	// create normal distributions (with mean = 0) for x, y and theta
	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);

	// prediction configurations of particle after time evolution by delta_t
	for(unsigned int i=0; i<num_particles; ++i){

			// update x, y and theta (noise is added later)
			if(fabs(yaw_rate) > 0.0001){ // for avoiding division by yaw_rate = 0

				double theta_p = particles[i].theta + yaw_rate * delta_t;
				particles[i].x += velocity/yaw_rate * (sin(theta_p) - sin(particles[i].theta));
				particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(theta_p));
				particles[i].theta += yaw_rate * delta_t;

			}
			else{

				particles[i].x += velocity * delta_t * cos(particles[i].theta);
				particles[i].y += velocity * delta_t * sin(particles[i].theta);

			}

			// add Gaussian noises
			particles[i].x += dist_x(gen);
			particles[i].y += dist_y(gen);
			particles[i].theta += dist_theta(gen);

	}


}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

	for(unsigned int i=0; i<observations.size(); ++i){

		// Distance between a given observation and the predicted measurment closest to it.
		// Initially set to a sufficiently large value.
		double dist_op_min = numeric_limits<double>::max();

		// To save the id of the predicted measurement closest to a given observation
		int id_closest = -1;

		for(unsigned int j=0; j<predicted.size(); ++j){

			// Compute the distance between a given observation and a given predicted measurement
			double dist_op = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			// If a given predicted measurement is the closest to the observation so far,
			// save the distance and the id of the predicted measurement.
			if(dist_op < dist_op_min){

				dist_op_min = dist_op;
				id_closest = predicted[j].id;

			}

		}

		// update of the id for the observation
		observations[i].id = id_closest;

	}

}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
			const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];

	// the followings are used when computing the multivariate normal distribution
	double denom_x = 2.0 * sig_x * sig_x;
	double denom_y = 2.0 * sig_y * sig_y;
	double gauss_norm = 1.0 / (2.0 * M_PI * sig_x * sig_y);

	for(unsigned int i=0; i< num_particles; ++i){

		vector<LandmarkObs> predicted;
		vector<LandmarkObs> obs_transformed;

		// if the distance between a given landmark and the particle is within the sensor range,
		// then save this landmark as a predicted measurement of landmarks
		for(unsigned int j=0; j< map_landmarks.landmark_list.size(); ++j){

			double x_l = map_landmarks.landmark_list[j].x_f;
			double y_l = map_landmarks.landmark_list[j].y_f;
			int id_l = map_landmarks.landmark_list[j].id_i;

			// distance between the landmark and particle
			double dist_lp = dist(x_l, y_l, particles[i].x, particles[i].y);

			if(dist_lp <= sensor_range){
				predicted.push_back(LandmarkObs{id_l, x_l, y_l});
			}

		}

		// transform the observations to map's coordinate system
		double theta_p = particles[i].theta;
		for(unsigned int j=0; j<observations.size(); ++j){

			double x_obs = particles[i].x
											+ observations[j].x * cos(theta_p) - observations[j].y * sin(theta_p);
			double y_obs = particles[i].y
											+ observations[j].x * sin(theta_p) + observations[j].y * cos(theta_p);

			obs_transformed.push_back(LandmarkObs{observations[j].id, x_obs, y_obs});

		}

		// update the assignment of the observations to the landmarks
		dataAssociation(predicted, obs_transformed);

		// update the weight of the particle
		particles[i].weight = 1.0; //initialization
		for(unsigned int j=0; j< obs_transformed.size(); ++j){

			int ind_obs = obs_transformed[j].id;

			// retrieve x, y coordinate of the closest landmark
			double mu_x = 0.0;
			double mu_y = 0.0;
			for(unsigned int k=0; k<predicted.size(); ++k){

				if(predicted[k].id == ind_obs){
					mu_x = predicted[k].x;
					mu_y = predicted[k].y;
					break;
				}

			}

			// compute the multivariate normal distribution and update the weight
			double x_diff = obs_transformed[j].x - mu_x;
			double y_diff = obs_transformed[j].y - mu_y;
			double exponent = x_diff * x_diff / denom_x + y_diff * y_diff / denom_y;
			particles[i].weight *= gauss_norm * exp((-1.0) * exponent);

		}

		weights[i] = particles[i].weight;

	}

}


void ParticleFilter::resample() {

	default_random_engine gen;

	vector<Particle> p_resampled(num_particles);

	// resample particles
	for(unsigned int i=0; i<num_particles; ++i){

		discrete_distribution<int> resampled_ind(weights.begin(), weights.end());
		p_resampled[i] = particles[resampled_ind(gen)];

	}

	// update particles
	particles = p_resampled;

}


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

		return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
