#include <iostream>  
#include <typeinfo>
#include "opencv2/opencv.hpp"  

using namespace cv;
using namespace std;

int main()
{
	// VideoCapture�� Video������ �ҷ���
	VideoCapture vc(0); if (!vc.isOpened())return 0; // connection failed
	vc.set(CV_CAP_PROP_FRAME_WIDTH, 640);  // CAP_PROP�� ����...
	vc.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	// �ʿ亯�� ����
	// Ķ���극�̼� ���
	cv::Mat K = (cv::Mat_<double>(3, 3) << 634.562348, 0, 305.212497, 0, 634.205282, 256.234517, 0, 0, 1);
	cv::Mat dist_coeff = (cv::Mat_<double>(4, 1) << 0.004628, -0.044073, 0.008047, 0.001567);
	Mat img;  // ���� �޾��� Matrix ����
	Size boardSize(4, 7); // chessboard ������ (üũ���� ���� ����)
	float board_cellsize = 26.0;
	// �������� ť�� ���� ����
	std::vector<cv::Point3d> box_lower, box_upper;
	box_lower.push_back(cv::Point3d(3 * board_cellsize, 2 * board_cellsize, 0));
	box_lower.push_back(cv::Point3d(5 * board_cellsize, 2 * board_cellsize, 0));
	box_lower.push_back(cv::Point3d(5 * board_cellsize, 4 * board_cellsize, 0));
	box_lower.push_back(cv::Point3d(3 * board_cellsize, 4 * board_cellsize, 0));
	box_upper.push_back(cv::Point3d(3 * board_cellsize, 2 * board_cellsize, -2 * board_cellsize));
	box_upper.push_back(cv::Point3d(5 * board_cellsize, 2 * board_cellsize, -2 * board_cellsize));
	box_upper.push_back(cv::Point3d(5 * board_cellsize, 4 * board_cellsize, -2 * board_cellsize));
	box_upper.push_back(cv::Point3d(3 * board_cellsize, 4 * board_cellsize, -2 * board_cellsize));


	//��ķ���� ĸ�ĵǴ� �̹��� ũ�⸦ ������
	Size size = Size((int)vc.get(CAP_PROP_FRAME_WIDTH),
		(int)vc.get(CAP_PROP_FRAME_HEIGHT));

	//���Ϸ� �������� �����ϱ� ���� �غ�  
	VideoWriter outputVideo;
	outputVideo.open("ouput.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'),
		30, size, true);
	if (!outputVideo.isOpened())
	{
		cout << "�������� �����ϱ� ���� �ʱ�ȭ �۾� �� ���� �߻�" << endl;
		return 1;
	}


	// �̹��� ó�� / ���� ����
	while (1){
		vc >> img; if (img.empty())break;  // vc�� �о���� ������ img�� ����
		flip(img, img, 1);				   // ���� ���ϰ� flip


		// vector�� �̹��� ��ǥ �޾��� ���� ����
		vector<std::vector<cv::Point2f> > imagePoints(1);
		vector<std::vector<cv::Point3f> > objectPoints(1);
		Mat rvec, tvec;
		bool found = findChessboardCorners(img, boardSize, imagePoints[0]);
		//printf("Hello?");
		for (int r = 0; r < boardSize.height; r++)
			for (int c = 0; c < boardSize.width; c++)
			// push_back (vector ���� �ڷ� �߰��ϴ� ��ɾ�)
				objectPoints[0].push_back(cv::Point3d(board_cellsize * c, board_cellsize * r, 0));


		if (found){ // ü�����带 ã���� �ϴ� �͵�
			// solvePnP�� rvec, tvec ã��!
			cv::solvePnP(objectPoints[0], imagePoints[0], K, dist_coeff, rvec, tvec); // objPoints�� 3��������
			drawChessboardCorners(img, boardSize, cv::Mat(imagePoints[0]), found);
		//	cout << typeid(tvec).name() << endl;
			Mat line_lower, line_upper;
			//  cv::projectPoints(������ ������ǥ, rVec, tVec, K(�������), distCoeffs, �����.��������ǥ)
			projectPoints(box_lower, rvec, tvec, K, dist_coeff, line_lower);
			projectPoints(box_upper, rvec, tvec, K, dist_coeff, line_upper);
			// reshape(����) : ä���� ���ڸ�ŭ �Ȱ��� ����, converTo(�������,�ڷ���):polyline�� ���� CV_32S�� �ٲ���. 
			line_lower.reshape(1).convertTo(line_lower, CV_32S); // Change 4 x 1 matrix (CV_64FC2) to 4 x 2 matrix (CV_32SC1)
			line_upper.reshape(1).convertTo(line_upper, CV_32S); // Because 'cv::polylines()' only accepts 'CV_32S' depth.
			//cout << line_lower << endl;
			//break;
			// polylines(��ȭ��,��ǥ��,t/f �����ĸ���, Į��(��Į���), shift)
			cv::polylines(img, line_lower, true, cv::Scalar(255, 0, 0), 2);
			for (int i = 0; i < line_lower.rows; i++){
				// cv::line(��ȭ��, ������,����,Scalar(Į��),(����׸�..? :�β�,����Ÿ��),shift)
				cv::line(img, cv::Point(line_lower.row(i)), cv::Point(line_upper.row(i)), cv::Scalar(0, 255, 255), 2); // �ذ� �� ����
				cout << "�̰� ����lower" << line_lower.row(i) << "�̰� ���� upper" <<line_upper.row(i)<< endl;
			}
			cv::polylines(img, line_upper, true, cv::Scalar(0, 0, 255), 2);

		}
		
		imshow("cam", img);
		//������ ���Ͽ� ���������� ������.  
		outputVideo << img;
		if (waitKey(10) == 27)break;
	}
	destroyAllWindows();
	return 0;
}
