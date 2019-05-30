#ifndef LAHBPCG_H
#define LAHBPCG_H

#include <Eigen/Core>
#include <vector>
#include <ctime>

class LAHBPCG {
	// Four off-diagonal elements
	struct S_elems {
		float SS;
		float SE;
		float SN;
		float SW;
	};

public:
	
	LAHBPCG();
	LAHBPCG(Eigen::MatrixXd& d, Eigen::MatrixXd& gx, Eigen::MatrixXd& gy, Eigen::MatrixXd& w, Eigen::MatrixXd& sx, Eigen::MatrixXd& sy);
	
	void solve(Eigen::MatrixXd& guess, int max_iter, float tol);
	Eigen::MatrixXd apply(Eigen::MatrixXd& d, Eigen::MatrixXd& gx, Eigen::MatrixXd& gy, Eigen::MatrixXd& w, Eigen::MatrixXd& sx,
		Eigen::MatrixXd& sy, int max_iter, float tol);
private:
	Eigen::MatrixXd Ax(Eigen::MatrixXd im); // apply A to "x" the Eigen::MatrixXd

	Eigen::MatrixXd hbPrecondition(Eigen::MatrixXd& r); // apply the preconditioner to the residual r

	void RBBmaps();
	void constructPreconditioner();
	void ind2xy(const unsigned int index, int & x, int & y);

	unsigned int varIndices(const int x, const int y);
	
	// these are just references to already allocated memory
	//Eigen::MatrixXd ADcoarse; 
	Eigen::MatrixXd AW;
	Eigen::MatrixXd AN;
	Eigen::MatrixXd w;
	Eigen::MatrixXd sx;
	Eigen::MatrixXd sy;

	Eigen::MatrixXd b; // const?

	Eigen::MatrixXd f; // current iterate storate....
	Eigen::MatrixXd hbRes;

	Eigen::MatrixXd AD; // diagonalized A matrix
	unsigned int max_length;

	std::vector< std::vector<unsigned int> > index_map; // goes up to 2^32
	std::vector< std::vector< S_elems > > S; // 4 channel Eigen::MatrixXds that store S weights...

};


#endif // !LAHBPCG_H
