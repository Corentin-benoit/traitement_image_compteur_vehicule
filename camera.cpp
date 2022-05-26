#include "camera.h"
#include <opencv2/highgui/highgui_c.h>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace cv;
using namespace std;

Camera::Camera()
{
	m_fps = 30;
}

bool Camera::open(string filename)
{
	m_fileName = filename;

	// Convert filename to number if you want to
	// open webcam stream
	istringstream iss(filename.c_str());
	int devid;
	bool isOpen;
	if(!(iss >> devid))
	{
		isOpen = m_cap.open(filename.c_str());
	}
	else
	{
		isOpen = m_cap.open(devid);
	}

	if(!isOpen)
	{
		cerr << "Unable to open video file." << endl;
		return false;
	}

	// set framerate, if unable to read framerate, set it to 30
	m_fps = m_cap.get(CAP_PROP_FPS);
	if(m_fps == 0)
		m_fps = 30;
}

void Camera::play()
{
	// Create main window
	//namedWindow("Video", WINDOW_AUTOSIZE);
	bool isReading = true;
	// Compute time to wait to obtain wanted framerate
	int timeToWait = 1000/m_fps;
	int count = 0;

	Mat bgReferenceGS;
	Mat bgReference;
	Mat carMask, m_gray;

	int first_image = 0;
	int first_image2 = 0;
	vector<Vec4i> m_road;
	Mat m_frame_initial;
	Mat m_Sgray, m_Sgray_initial, m_Sbinarize, m_Sdiff, m_Sclose, m_Sopen, m_Sdilate, m_Serode, m_Scanny, m_Sgauss;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	int count_droite = 0;
	int count_gauche = 0;
	Scalar color = Scalar(0, 0, 255 );


	while(isReading)
	{
		// Get frame from stream
		isReading = m_cap.read(m_frame);


		if (count == 3) {
			// Obtenir l'image de fond de référence
			bgReference = m_frame.clone();
			cvtColor(bgReference, bgReferenceGS, COLOR_BGR2GRAY);
		}

		if(isReading)
		{
			/*-----------------------------------------------------*/
			/*-----------------DETECTION DE LIGNE------------------*/
			/*-----------------------------------------------------*/

			// Création de matrice et de vecteur
			Mat m_hsv, m_filtered, m_filtered_inv, m_opened, m_closed, m_edge;


			//Codé sur 8 bits donc 0 < S,V < 255
			// 0 < H < 360
			int low_H = 20;
			int low_S = 25;
			int low_V = 25;
			int high_H = 85;
			int high_S = 255;
			int high_V = 255;

			//Copie de la vidéo principale pour pouvoir écrire dessus
			Mat m_frame_final = m_frame.clone();

			if (first_image < 2)
			{
				first_image++;
				cout << first_image << endl;

				// Conversion de couleurs de RGB à HSV
				cvtColor(m_frame, m_hsv, COLOR_BGR2HSV);
				//imshow("HSV", m_hsv);

				//Vérifie si les éléments d'un tableau sont situés entre les éléments de deux autres tableaux
				//en particulier, avec ce cone HSV on sélectionne la couleur de l'herbe
				inRange(m_hsv, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), m_filtered);
				//imshow("Cône HSV", m_filtered);

				//Inversion des pixels pour ne faire apparaitre que la route
				bitwise_not(m_filtered, m_filtered_inv);
				//imshow("Inversion cône HSV", m_filtered_inv);

				//On réalise une fermeture
				morphologyEx(m_filtered_inv, m_closed, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)));
				//imshow("Fermeture", m_closed);

				//On réalise une ouverture
				morphologyEx(m_closed, m_opened, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(12, 12)));
				//imshow("Ouverture", m_opened);

				// On trace les bords
				Canny(m_opened, m_edge, 0, 0, 3);
				//imshow("Bordure", m_edge);

				// Ne conserve que les lignes droites
				HoughLinesP(m_edge, m_road, 1, 3.14/180, 60, 100, 100);
				//imshow("Bordure", m_edge);


				//imshow("Image finale", m_frame_final);
			}

			//Trace les lignes
			for (size_t i = 0; i < m_road.size(); i++) {
				Vec4i l = m_road[i];
				line(m_frame_final, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
			}

			imshow("Video avec lignes", m_frame_final);

			/*-----------------------------------------------------*/
			/*-----------------SUIVI DES VEHICULES-----------------*/
			/*-----------------------------------------------------*/


			Mat m_Sframe_compt = m_frame.clone();

			if(first_image2 == 1){
				m_frame_initial = m_frame.clone();
				cvtColor(m_frame_initial, m_Sgray_initial, COLOR_BGR2GRAY);
				//imshow("V - Image binarisée", m_Sbinarize_initial);
			}
			first_image2++;


			if(first_image2 > 1){

				// On binarise l'image
				cvtColor(m_frame, m_Sgray, COLOR_BGR2GRAY);
				//imshow("V - Video binarisée", m_Sbinarize);

				//Réalise la différence entre l'image témoin et la vidéo
				absdiff(m_Sgray_initial, m_Sgray, m_Sdiff);
				//imshow("V - Video diff", m_Sdiff);

				//On floute
				medianBlur(m_Sdiff, m_Sgauss,5);
				//imshow("V - Video flou", m_Sgauss);

				//On binarise
				threshold(m_Sgauss, m_Sbinarize, 12, 255, THRESH_BINARY);
				//imshow("V - Video binarize", m_Sbinarize);

				//On réalise une ouverture
				morphologyEx(m_Sbinarize, m_Sopen, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(4,4)));
				//imshow("V - Video Ouverture", m_Sopen);

				//On dilate
				morphologyEx(m_Sopen, m_Sdilate, MORPH_DILATE, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
				//imshow("V - Video dilate", m_Sdilate);

				//On ferme
				morphologyEx(m_Sdilate, m_Sclose, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
				//imshow("V - Video fermeture", m_Sclose);

				//On erode --> Pas utile
				//morphologyEx(m_Sclose, m_Serode, MORPH_ERODE, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
				//imshow("V - Video erosion", m_Serode);


				//On détecte les bords
				Canny( m_Sclose, m_Scanny, 100 , 200);
				imshow("V - Video canny", m_Scanny);

				//On en déduit les contours
				findContours(m_Scanny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );


				Mat drawing = Mat::zeros( m_Scanny.size(), CV_8UC3 );

				//On trace un carré rouge autour des vehicules
				vector<vector<Point> > contours_poly( contours.size() );
				vector<Rect> boundRect( contours.size() );
				for( size_t i = 0; i < contours.size(); i++ )
				{
					approxPolyDP( contours[i], contours_poly[i], 3, true );
					boundRect[i] = boundingRect( contours_poly[i]);

					//drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );

					int x = boundRect[i].x + boundRect[i].width/2;
					int y = boundRect[i].y + boundRect[i].height/2;

					// On réculpère seulement les contours avec un périmètre assez important
					if(arcLength(contours[i], true) > 120){

						// On centre le carré rouge
						Point center;
						center.x = x;
						center.y = y;
						float range = 2;

						//Compteur pour les voies de droite et de gauche avec un compteur principal
						int* count = &count_gauche;
						if (center.x > m_frame.size().width / 2)
						 	count = &count_droite;

						if(center.y <= (120 + range) && center.y >= (120 - range))
						{
							*count += 1;
							cout << "gauche = "<< count_gauche << " droite ="<< count_droite << endl;
						}

						circle(m_frame_final, center, 5, Scalar(255,60,0),-1);

						rectangle(m_frame_final, boundRect[i].tl(), boundRect[i].br(), color, 2 );
					}
				}
				line(m_frame_final, Point(10, 110), Point(240, 140), Scalar(255,60,0), 3, LINE_AA);
				line(m_frame_final, Point(520, 130), Point(710, 90), Scalar(255,60,0), 3, LINE_AA);
				imshow( "Contours", m_frame_final );
			}
			/*
			if((center.x*40/170)+64 <= (center.y + range) && (center.x*40/170)+64 >= (center.y - range))
						{
							*count += 1;
							cout << "gauche = "<< count_gauche << " droite ="<< count_droite << endl;
						}
						else if((-center.x/3)+316 <= (center.y + range) && (-center.x/3)+316 >= (center.y - range))
						{
							*count += 1;
							cout << "gauche = "<< count_gauche << " droite ="<< count_droite << endl;
						}
*/


		}
		else
		{
			cerr << "Unable to read device" << endl;
		}

		// If escape key is pressed, quit the program
		if(waitKey(timeToWait)%256 == 27)
		{
			cerr << "Stopped by user" << endl;
			isReading = false;
		}
		count++;
	}
}

bool Camera::close()
{
	// Close the stream
	m_cap.release();

	// Close all the windows
	destroyAllWindows();
	usleep(100000);
}
