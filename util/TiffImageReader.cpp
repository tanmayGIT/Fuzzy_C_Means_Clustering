//
//  TiffImageReader.cpp
//  DocScanImageProcessing
//
//  Created by Tanmoy on 7/2/17.
//  Copyright © 2017 Tanmoy. All rights reserved.
//

#include "TiffImageReader.hpp"


/*
 This is the library for reading the tiff images. The opencv library was not working to read tiff images, so I need to have this code for reading the tiff images.
 */
//void TiffImageReader::readTiffFiles(string imgName)
//{
//    string imageName(imgName); // start with a default
//    // Open the TIFF file using libtiff
//    libtiff::TIFF* tif = libtiff::TIFFOpen(imageName.c_str(), "r");
//    // Create a matrix to hold the tif image in
//    cv::Mat image;
//    // check the tif is open
//    if (tif) {
//        do {
//            unsigned int width, height;
//            libtiff::uint32* raster;
//            
//            // get the size of the tiff
//            TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
//            TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
//            
//            uint npixels = width*height; // get the total number of pixels
//            
//            raster = (libtiff::uint32*)libtiff::_TIFFmalloc(npixels * sizeof(libtiff::uint32)); // allocate temp memory (must use the tiff library malloc)
//            if (raster == NULL) // check the raster's memory was allocaed
//            {
//                TIFFClose(tif);
//                assert("Could not allocate memory for raster of TIFF image");
//            }
//            
//            // Check the tif read to the raster correctly
//            if (!TIFFReadRGBAImage(tif, width, height, raster, 0))
//            {
//                TIFFClose(tif);
//                assert("Could not read raster of TIFF image");
//            }
//            
//            image = cv::Mat(width, height, CV_8UC4); // create a new matrix of w x h with 8 bits per channel and 4 channels (RGBA)
//            
//            // itterate through all the pixels of the tif
//            for (uint x = 0; x < width; x++)
//                for (uint y = 0; y < height; y++)
//                {
//                    libtiff::uint32& TiffPixel = raster[y*width+x]; // read the current pixel of the TIF
//                    cv::Vec4b& pixel = image.at<cv::Vec4b>(cv::Point(y, x)); // read the current pixel of the matrix
//                    pixel[0] = TIFFGetB(TiffPixel); // Set the pixel values as BGRA
//                    pixel[1] = TIFFGetG(TiffPixel);
//                    pixel[2] = TIFFGetR(TiffPixel);
//                    pixel[3] = TIFFGetA(TiffPixel);
//                }
//            
//            libtiff::_TIFFfree(raster); // release temp memory
//            // Rotate the image 90 degrees couter clockwise
//            image = image.t();
//            flip(image, image, 0);
//            imshow("TIF Image", image); // show the image
//            cv::waitKey(0); // wait for anykey before displaying next
//        } while (TIFFReadDirectory(tif)); // get the next tif
//        TIFFClose(tif); // close the tif file
//    }
//}
//
//
