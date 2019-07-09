#include "LAHBPCG.h"

LAHBPCG::LAHBPCG() {

}

LAHBPCG::LAHBPCG(Eigen::MatrixXd& d, Eigen::MatrixXd& gx, Eigen::MatrixXd& gy, Eigen::MatrixXd& w, Eigen::MatrixXd& sx, Eigen::MatrixXd& sy) {

	this->AW = Eigen::MatrixXd(d.rows(), d.cols());
	this->AN = Eigen::MatrixXd(d.rows(), d.cols());
	this->AD = Eigen::MatrixXd(d.rows(), d.cols());
	this->w = w;
	this->sx = sx;
	this->sy = sy;
	this->b = Eigen::MatrixXd(d.rows(), d.cols());
	this->f = Eigen::MatrixXd(d.rows(), d.cols());
	this->hbRes = Eigen::MatrixXd(d.rows(), d.cols());
	max_length = d.rows() * d.cols();
	
	float sub_y, sub_x;
	float add_y, add_x;
	for (int y = 0; y < b.cols(); y++) {
		for (int x = 0; x < b.rows(); x++) {

			if (y == b.cols() - 1) {
				add_y = 0;
			}
			else {
				add_y = sy(x, y + 1);
			}
			
			if (x == b.rows() - 1) {
				add_x = 0;
			}
			else {
				add_x = sx(x + 1, y);
			}
			
			AD(x, y) = sx(x, y) + add_x + w(x, y) + sy(x, y) + add_y;

			if (y == b.cols() - 1) {
				AN(x, y) = 0;
			}
			else {
				AN(x, y) = -sy(x, y + 1);
			}
			
			if (x == b.rows() - 1) {
				AW(x, y) = 0;
			}
			else {
				AW(x, y) = -sx(x + 1, y);
			}
			
			if (y == b.cols() - 1) {
				sub_y = 0;
			}
			else {
				sub_y = gy(x, y + 1) * sy(x, y + 1);
			}

			if (x == b.rows() - 1) {
				sub_x = 0;
			}
			else {
				sub_x = gx(x + 1, y) * sx(x + 1, y);
			}
			
			b(x, y) = (gy(x, y) * sy(x, y) - sub_y + gx(x, y) * sx(x, y) - sub_x + w(x, y) * d(x, y));

		}

	}

	RBBmaps();
	constructPreconditioner();

}

// computes the vector indices into an image (0 to m*n - 1)
void LAHBPCG::RBBmaps() {

	int numOctaves = ceil(logf(std::min(f.cols(), f.rows())) / logf(2.0f));
	int by = 0; int bx = 0; // (0,0) is black

	int a = 1;
	std::vector<unsigned int> valid;
	std::vector<unsigned int> newValid;

	for (int i = 0; i < numOctaves; i++) {

		std::vector<unsigned int> indices1;
		std::vector<unsigned int> indices2;

		//indices1.reserve(f.width * f.height);
		//indices2.reserve(f.width * f.height);

		if (valid.empty()) { // loop over all elements
			for (int x = 0; x < f.rows(); x++) {
				for (int y = 0; y < f.cols(); y++) {

					if ((x + y) % (2 * a) == (by + bx + a) % (2 * a)) {
						indices1.push_back(x * f.cols() + y);
					}
					else {
						valid.push_back(x * f.cols() + y);
					}

				}
			}

			if (indices1.size() == 0) {
				break;
			}
			index_map.push_back(indices1);

			int x, y;

			newValid.clear();
			for (std::vector<unsigned int>::iterator it = valid.begin(); it != valid.end(); ++it) {
				ind2xy(*it, x, y);
				if (y % (2 * a) == (by + a) % (2 * a)) {
					indices2.push_back(*it);
				}
				else {
					newValid.push_back(*it);
				}
			}
			valid.swap(newValid);

			if (indices2.size() == 0) {
				break;
			}
			index_map.push_back(indices2);

		}
		else { // iterate over list elements (containing valid indices)
			int x, y;

			newValid.clear();
			for (std::vector<unsigned int>::iterator it = valid.begin(); it != valid.end(); ++it) {
				ind2xy(*it, x, y);
				if ((x + y) % (2 * a) == (bx + by + a) % (2 * a)) {
					indices1.push_back(*it);
				}
				else {
					newValid.push_back(*it);
				}
			}
			valid.swap(newValid);

			if (indices1.size() == 0) {
				break;
			}
			index_map.push_back(indices1);

			newValid.clear();
			for (std::vector<unsigned int>::iterator it = valid.begin(); it != valid.end(); ++it) {
				ind2xy(*it, x, y);
				if ((y) % (2 * a) == (by + a) % (2 * a)) {
					indices2.push_back(*it);
				}
				else {
					newValid.push_back(*it);
				}
			}
			valid.swap(newValid);

			if (indices2.size() == 0) {
				break;
			}
			index_map.push_back(indices2);
		}

		a *= 2;
	}
}

// computes the x,y index from a vector index
void LAHBPCG::ind2xy(const unsigned int index, int & x, int & y) {
	x = index / f.cols();
	y = index % f.cols();
}

// computes the preconditioner
void LAHBPCG::constructPreconditioner() {
	
	int x, y, x1, y1;
	// fill out S matrix
	for (int k = 0; k < (int)index_map.size(); k++) {

		bool oddLevel = (k + 1) % 2;
		int stride = 1 << (k / 2);
		//printf("stride: %d\n", stride);

		unsigned int dn1, dn2;
		if (oddLevel) {
			dn1 = stride; dn2 = stride * f.cols();
		}
		else {
			dn1 = stride*(f.cols() - 1);
			dn2 = stride*(f.cols() + 1);
		}

		// compute S elements at this level
		std::vector< S_elems > S_elem_vec;
		std::vector< float > AD_old; // retain old values of AD

								//printf("dn1 %d, dn2 %d\n", dn1, dn2);
		S_elems elems;// = {0, 0, 0, 0};
					  // on this level, we use the indices in index_map[k]        
		for (std::vector<unsigned int>::iterator idx = index_map[k].begin(); idx != index_map[k].end(); ++idx) { 
			//int x, y, x1, y1;// x2, y2;

			ind2xy(*idx, x, y);

			elems.SS = -AN(x, y) / (AD(x, y)); // SS
			elems.SE = -AW(x, y) / (AD(x, y)); // SE

			if (*idx < dn1) { // *idx - dn1 < 0
				ind2xy(max_length + (*idx) - dn1, x1, y1);
			}
			else {
				ind2xy((*idx - dn1) % max_length, x1, y1);
			}
			elems.SN = -AN(x1, y1) / (AD(x, y)); // SN

			if (*idx < dn2) { // *idx - dn2 < 0
				ind2xy(max_length + (*idx) - dn2, x1, y1);
			}
			else {
				ind2xy((*idx - dn2) % max_length, x1, y1);
			}
			elems.SW = -AW(x1, y1) / (AD(x, y)); // SW

			S_elem_vec.push_back(elems);
			AD_old.push_back(AD(x, y));
			//AN(x1,y1)*elems.SN
			//AW(x2,y2)*elems.SW
			//printf("ind: %d, SW: %f\n", *idx, elems.SW);
		} /* end vector iterator */
		S.push_back(S_elem_vec);

		// now we need to redistribute edges....
		// need temp storage for this new AN....
		Eigen::MatrixXd AN_tmp(AN.rows(), AN.cols()); // actually "sparse"
		Eigen::MatrixXd AW_tmp(AW.rows(), AW.cols()); // actually "sparse"

		int i = 0;
		// modify AD at this level
		for (std::vector<unsigned int>::iterator idx = index_map[k].begin(); idx != index_map[k].end(); ++idx) {

			S_elems elems = S_elem_vec[i];
			ind2xy(*idx, x, y);

			ind2xy((*idx - dn1) % max_length, x1, y1);
			AD(x1, y1) += AN(x1, y1) * elems.SN;

			ind2xy((*idx - dn2) % max_length, x1, y1);
			AD(x1, y1) += AW(x1, y1) * elems.SW;

			ind2xy((*idx + dn1) % max_length, x1, y1);
			AD(x1, y1) += AN(x, y) * elems.SS;

			ind2xy((*idx + dn2) % max_length, x1, y1);
			AD(x1, y1) += AW(x, y) * elems.SE;

			/* end modify AD */

			/* now eliminate connections */
			unsigned int n_ind, w_ind, s_ind, e_ind;
			if (oddLevel) {
				n_ind = varIndices(x, y - stride);
				w_ind = varIndices(x - stride, y);
				s_ind = varIndices(x, y + stride);
				e_ind = varIndices(x + stride, y);
			}
			else {
				n_ind = varIndices(x - stride, y - stride);
				w_ind = varIndices(x - stride, y + stride);
				s_ind = varIndices(x + stride, y + stride);
				e_ind = varIndices(x + stride, y - stride);
			}
			//printf("n: %d, w: %d, s: %d, e: %d\n", n_ind, w_ind, s_ind, e_ind);

			// eliminate NS connections
			bool ns = false;
			float ns_weight = 0.0f;
			int n_x, n_y, s_x, s_y;

			ind2xy(n_ind, n_x, n_y);
			ind2xy(s_ind, s_x, s_y);
			if (n_ind < max_length && s_ind < max_length) {
				ns = true;
				if (oddLevel) {
					ns_weight = -AD_old[i] * elems.SN*elems.SS;
				}
				else {
					ns_weight = -AD_old[i] * elems.SW*elems.SE;
				}

				AD(n_x, n_y) += ns_weight;
				AD(s_x, s_y) += ns_weight;
				//printf("ns_weight: %f\n", ns_weight);
			}

			// eliminate WE connections
			bool we = false;
			float we_weight = 0.0f;
			int w_x, w_y, e_x, e_y;

			ind2xy(w_ind, w_x, w_y);
			ind2xy(e_ind, e_x, e_y);
			if (w_ind < max_length && e_ind < max_length) {
				we = true;
				if (oddLevel) {
					we_weight = -AD_old[i] * elems.SW*elems.SE;
				}
				else {
					we_weight = -AD_old[i] * elems.SN*elems.SS;
				}

				AD(w_x, w_y) += we_weight;
				AD(e_x, e_y) += we_weight;
				//printf("we_weight: %f\n", we_weight);
			}

			// redistribute "connected" edges / weights
			float nw_weight, ws_weight, se_weight, en_weight;
			nw_weight = ws_weight = se_weight = en_weight = 0.0f;
			if (oddLevel) {
				if (n_ind < max_length && w_ind < max_length) {
					nw_weight = AD_old[i] * elems.SN*elems.SW;
					AN_tmp(w_x, w_y) -= nw_weight;
				}
				if (w_ind < max_length && s_ind < max_length) {
					ws_weight = AD_old[i] * elems.SW*elems.SS;
					AW_tmp(w_x, w_y) -= ws_weight;
					//printf("nw_weight: %f\n", nw_weight);
				}
				if (e_ind < max_length && s_ind < max_length) {
					se_weight = AD_old[i] * elems.SE*elems.SS;
					AN_tmp(s_x, s_y) -= se_weight;
					//printf("nw_weight: %f\n", nw_weight);
				}
				if (e_ind < max_length && n_ind < max_length) {
					en_weight = AD_old[i] * elems.SE*elems.SN;
					AW_tmp(n_x, n_y) -= en_weight;
					//printf("nw_weight: %f\n", nw_weight);
				}

			}
			else {
				if (n_ind < max_length && w_ind < max_length) {
					nw_weight = AD_old[i] * elems.SN*elems.SW;
					AN_tmp(n_x, n_y) -= nw_weight;
					//printf("n_ind: %d, ind: %d, ni_weight: %f, wi_weight: %f\n", n_ind, *idx, elems.SW, elems.SN); // how to index into S...?

					//printf("nw_weight: %f\n", nw_weight);
				}
				if (w_ind < max_length && s_ind < max_length) {
					ws_weight = AD_old[i] * elems.SN*elems.SE;
					AW_tmp(w_x, w_y) -= ws_weight;
					//printf("nw_weight: %f\n", nw_weight);
				}
				if (e_ind < max_length && s_ind < max_length) {
					se_weight = AD_old[i] * elems.SE*elems.SS;
					AN_tmp(e_x, e_y) -= se_weight;
					//printf("nw_weight: %f\n", nw_weight);
				}
				if (e_ind < max_length && n_ind < max_length) {
					en_weight = AD_old[i] * elems.SS*elems.SW;
					AW_tmp(n_x, n_y) -= en_weight;
					//printf("nw_weight: %f\n", nw_weight);
				}
			}

			// normalize the redistributed weights
			if (ns || we) {
				float total = nw_weight + ws_weight + se_weight + en_weight;
				if (total != 0) {
					nw_weight /= total;
					ws_weight /= total;
					se_weight /= total;
					en_weight /= total;
				}

				// now, redistribute
				static const float sN = 2;
				float distWeight = sN*(ns_weight + we_weight);

				//printf("nw_weight %f, ws_weight %f, se_weight %f, en_weight %f, distWeight %f\n", nw_weight, ws_weight, se_weight, en_weight, distWeight);
				if (oddLevel) {
					if (n_ind < max_length && w_ind < max_length) {
						AN_tmp(w_x, w_y) += nw_weight*distWeight;
					}
					if (w_ind < max_length && s_ind < max_length) {
						AW_tmp(w_x, w_y) += ws_weight*distWeight;
					}
					if (e_ind < max_length && s_ind < max_length) {
						AN_tmp(s_x, s_y) += se_weight*distWeight;
					}
					if (e_ind < max_length && n_ind < max_length) {
						AW_tmp(n_x, n_y) += en_weight*distWeight;
					}
				}
				else {
					if (n_ind < max_length && w_ind < max_length) {
						AN_tmp(n_x, n_y) += nw_weight*distWeight;
					}
					if (w_ind < max_length && s_ind < max_length) {
						AW_tmp(w_x, w_y) += ws_weight*distWeight;
					}
					if (e_ind < max_length && s_ind < max_length) {
						AN_tmp(e_x, e_y) += se_weight*distWeight;
					}
					if (e_ind < max_length && n_ind < max_length) {
						AW_tmp(n_x, n_y) += en_weight*distWeight;
					}
				}

				if (n_ind < max_length && w_ind < max_length) {
					AD(n_x, n_y) -= nw_weight*distWeight;
					AD(w_x, w_y) -= nw_weight*distWeight;
				}
				if (w_ind < max_length && s_ind < max_length) {
					AD(w_x, w_y) -= ws_weight*distWeight;
					AD(s_x, s_y) -= ws_weight*distWeight;
				}
				if (e_ind < max_length && s_ind < max_length) {
					AD(s_x, s_y) -= se_weight*distWeight;
					AD(e_x, e_y) -= se_weight*distWeight;
				}
				if (e_ind < max_length && n_ind < max_length) {
					AD(n_x, n_y) -= en_weight*distWeight;
					AD(e_x, e_y) -= en_weight*distWeight;
				}

			}

			i++; // go to next element to "eliminate"

		} /* end vector iterator */

		AN = AN_tmp;
		AW = AW_tmp;
	} /* end for all levels */

}

// applies the sparse, pentadiagonal matrix A to x (stored in im)
// assumes gradient images taken from ImageStack's gradient operator
// (i.e. backward differences) if not, results could be bogus!
Eigen::MatrixXd LAHBPCG::Ax(Eigen::MatrixXd im) {

	float a1, a2, a3;

	// (Ax + w)* x
	for (int y = 0; y < im.cols(); y++) {
		int x = 0;

		a1 = 0;
		a2 = sx(x, y) + sx(x + 1, y) + w(x, y);
		a3 = -sx(x + 1, y);
		f(x, y) = a2*im(x, y) + a3*im(x + 1, y);
		
		for (x = 1; x < im.rows() - 1; x++) {
			a1 = -sx(x, y);
			a2 = sx(x, y) + sx(x + 1, y) + w(x, y); // a_(ij)x_(ij) + w_(ij)*x_(ij)
			a3 = -sx(x + 1, y); // AW
			f(x, y) = a1*im(x - 1, y) + a2*im(x, y) + a3*im(x + 1, y);
		}

		x = im.rows() - 1;
		a1 = -sx(x, y);
		a2 = sx(x, y) + w(x, y);
		a3 = 0; //AW
		f(x, y) = a1*im(x - 1, y) + a2*im(x, y);
	
	}

	// Ay*x + (Ax + w)*x
	for (int x1 = 0; x1 < im.rows(); x1 += 8) {

		int y = 0;
		for (int x = x1; (x < x1 + 8) && (x < im.rows()); x++) {
			a1 = 0;
			a2 = sy(x, y) + sy(x, y + 1);
			a3 = -sy(x, y + 1); //AN
			f(x, y) += a2*im(x, y) + a3*im(x, y + 1);
		}

		for (y = 1; y < im.cols() - 1; y++) {
			for (int x = x1; (x < x1 + 8) && (x < im.rows()); x++) {

				a1 = -sy(x, y);
				a2 = sy(x, y) + sy(x, y + 1);
				a3 = -sy(x, y + 1);
				f(x, y) += a1*im(x, y - 1) + a2*im(x, y) + a3*im(x, y + 1);
			
			}
		}

		y = im.cols() - 1;
		for (int x = x1; (x < x1 + 8) && (x < im.rows()); x++) {
			a1 = -sy(x, y);
			a2 = sy(x, y);
			a3 = 0; //AN
			f(x, y) += a1*im(x, y - 1) + a2*im(x, y);
		}
	}

	return f;

}

// apply the preconditioner to the residual r
Eigen::MatrixXd LAHBPCG::hbPrecondition(Eigen::MatrixXd& r) {
	// wonder if there's a way to apply the preconditioner in a cache coherent manner....
	hbRes = r; // ugh.. another deep copy... (no way around it...)
	int x, y, x1, y1;
	for (int k = 0; k < (int)index_map.size(); k++) {
		//int i = 0;
		// turns out that recomputing these numbers is 
		// faster than accessing it in memory because of the misses
		bool oddLevel = (k + 1) % 2;
		int stride = 1 << (k / 2);
		//printf("stride: %d\n", stride);

		unsigned int dn1, dn2;
		if (oddLevel) {
			dn1 = stride; dn2 = stride * f.cols();
		}
		else {
			dn1 = stride*(f.cols() - 1);
			dn2 = stride*(f.cols() + 1);
		}

		int i = 0;
		// S'*d
		for (std::vector<unsigned int>::iterator idx = index_map[k].begin(); idx != index_map[k].end(); ++idx) {
			// since this is S'*d, the index map represents "column" now
			//int x, y, x1, y1, x2, y2;
			S_elems elems = S[k][i];

			ind2xy(*idx, x, y);

			if (*idx + dn1 < max_length) {
				//elems.SS = 0;
				ind2xy((*idx + dn1) % max_length, x1, y1);
				//y1 %= sy.height; x1 %= sy.width;
				hbRes(x1, y1) += hbRes(x, y) * elems.SS;
			}

			if (*idx + dn2 < max_length) {
				ind2xy((*idx + dn2) % max_length, x1, y1);
				//y1 %= sx.height; x1 %= sx.width;
				hbRes(x1, y1) += hbRes(x, y) * elems.SE;
			}

			if (*idx >= dn1) { // *idx - dn1 >= 0
				ind2xy((*idx - dn1) % max_length, x1, y1);
				//y1 %= sy.height; x1 %= sy.width;
				hbRes(x1, y1) += hbRes(x, y) * elems.SN;
			}

			if (*idx >= dn2) { // *idx - dn2 >= 0
				ind2xy((*idx - dn2) % max_length, x1, y1);
				//y1 %= sx.height; x1 %= sx.width;
				hbRes(x1, y1) += hbRes(x, y) * elems.SW;
			}

			i++;
		}
	}

	hbRes.cwiseQuotient(AD);
	//Divide::apply(hbRes, AD); // invert the diagonal

	// lowest level is identity matrix so it's ignored (not even stored in index_map)
	for (int k = (int)index_map.size() - 1; k >= 0; k--) {
		//int i = index_map.size() - 1;
		bool oddLevel = (k + 1) % 2;
		int stride = 1 << (k / 2);
		//printf("stride: %d\n", stride);

		unsigned int dn1, dn2;
		if (oddLevel) {
			dn1 = stride;
			dn2 = stride * f.cols();
		}
		else {
			dn1 = stride * (f.cols() - 1);
			dn2 = stride * (f.cols() + 1);
		}

		int i = 0;
		//S*d
		for (std::vector<unsigned int>::iterator idx = index_map[k].begin(); idx != index_map[k].end(); ++idx) {
			// since this is S*d, the index map represents "row" now
			//int x, y, x1, y1, x2, y2;
			S_elems elems = S[k][i];

			ind2xy(*idx, x, y);

			if (*idx + dn1 < max_length) {
				ind2xy((*idx + dn1) % max_length, x1, y1);
				//y1 %= sy.height; x1 %= sy.width;
				hbRes(x, y) += hbRes(x1, y1) * elems.SS;
			}

			if (*idx + dn2 < max_length) {
				ind2xy((*idx + dn2) % max_length, x1, y1);
				//y1 %= sx.height; x1 %= sx.width;
				hbRes(x, y) += hbRes(x1, y1) * elems.SE;
			}

			if (*idx >= dn1) { // *idx - dn1 >= 0
				ind2xy((*idx - dn1) % max_length, x1, y1);
				//y1 %= sy.height; x1 %= sy.width;
				hbRes(x, y) += hbRes(x1, y1) * elems.SN;
			}

			if (*idx >= dn2) { // *idx - dn2 >= 0
				ind2xy((*idx - dn2) % max_length, x1, y1);
				//y1 %= sx.height; x1 %= sx.width;
				hbRes(x, y) += hbRes(x1, y1) * elems.SW;
			}

			i++;
		}
		//i--;
	}

	return hbRes;

}

// solve the PCG!
void LAHBPCG::solve(Eigen::MatrixXd& guess, int max_iter, float tol) {

	Eigen::MatrixXd dr_tmp, s;

	Eigen::MatrixXd r = b; // we currently do not use b anywhere else, so i reuse its memory
	//Subtract::apply(r, Ax(guess));
	r = r - Ax(guess);
	Eigen::MatrixXd dr = hbPrecondition(r); // precondition, dr to differentiate from d

	Eigen::VectorXd vR = Eigen::Map<Eigen::VectorXd>(const_cast<double *>(r.data()), r.rows() * r.cols(), 1);
	Eigen::VectorXd vDr = Eigen::Map<Eigen::VectorXd>(const_cast<double *>(dr.data()), dr.rows() * dr.cols(), 1);

	float delta = vR.dot(vDr);
	float epsilon = tol*tol*delta;
	
	for (int i = 1; i <= max_iter; i++) {
		if (delta < epsilon) {
			break;
		}

		vDr = Eigen::Map<Eigen::VectorXd>(const_cast<double *>(dr.data()), dr.rows() * dr.cols(), 1);
		Eigen::MatrixXd wr = Ax(dr);
		Eigen::VectorXd vWr = Eigen::Map<Eigen::VectorXd>(const_cast<double *>(wr.data()), wr.rows() * wr.cols(), 1);

		float alpha = delta / vDr.dot(vWr);

		dr_tmp = dr;

		dr = dr * alpha;
		guess = guess + dr;
		wr = wr * alpha;
		r = r - wr;
		//Scale::apply(dr, alpha);
		//Add::apply(guess, dr);    // guess = guess + alpha*dr
		//Scale::apply(wr, alpha);
		//Subtract::apply(r, wr);   // r = r - alpha*wr

		vR = Eigen::Map<Eigen::VectorXd>(const_cast<double *>(r.data()), r.rows() * r.cols(), 1);
		float resNorm = vR.dot(vR);
		//printf("iteration %d, error %f\n", i, resNorm);
		if (resNorm < epsilon) {
			break;
		}

		s = hbPrecondition(r);    // precondition
		Eigen::VectorXd vS = Eigen::Map<Eigen::VectorXd>(const_cast<double *>(s.data()), s.rows() * s.cols(), 1);
		float delta_old = delta;
		delta = vR.dot(vS);
		float beta = delta / delta_old;

		dr_tmp = dr_tmp * beta;
		//Scale::apply(dr_tmp, beta);
		dr = s;
		dr = dr + dr_tmp;
		//Add::apply(dr, dr_tmp);
	}
}

Eigen::MatrixXd LAHBPCG::apply(Eigen::MatrixXd& d, Eigen::MatrixXd& gx, Eigen::MatrixXd& gy, Eigen::MatrixXd& w, Eigen::MatrixXd& sx,
	Eigen::MatrixXd& sy, int max_iter, float tol)
{
	// check to make sure have same # of frames and same # of channels
	// assumes gradient images computed using ImageStack's gradient, which is
	// slightly different from the standard convolution gradient
	/*
		Image gx1(d);
		Image gy1(d);
		Gradient::apply(gx1, 'x');
		Gradient::apply(gy1, 'y');
	*/

	// runtime is LINEAR in the number of pixels!

	// solve the problem
	// minimize
	//  sum_(i,j) w_(i,j)*(f_(i,j) - d_(i,j))^2 + sum_(i,j) sx_i (f_(i+1,j) - f_(i,j) - gx(i,j))^2
	//    + sy_j (f_(i,j+1) - f(i,j) - gy(i,j))^2
	// which can be written as
	//
	// minimize x^T A x + b^T x + c
	//
	// the problem reduces to solving Ax = b
	
	Eigen::MatrixXd out(d.rows(), d.cols());

	// solves frames independently
	

	printf("Computing preconditioner...\n");
	LAHBPCG solver(d, gx, gy, w, sx, sy);

	printf("Solving...\n");
	solver.solve(out, max_iter, tol);

	return out;
}

unsigned int LAHBPCG::varIndices(const int x, const int y)
{

	if (x < 0 || x >= f.rows())
		return max_length;
	if (y < 0 || y >= f.cols())
		return max_length;
	return (x * f.cols() + y);

}
