import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;
import org.opencv.core.Scalar;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Point;
import org.openftc.easyopencv.OpenCvPipeline;
import java.util.List;
import java.util.ArrayList;

public class DuckPipeline extends OpenCvPipeline {

    static int position = 0;

    @Override
    public Mat processFrame(Mat inputMat) {
        //Clone the inputMat
        Mat outputMat = inputMat.clone();

        //Filter out colors that are not yellow
        Imgproc.cvtColor(inputMat, inputMat, Imgproc.COLOR_RGB2HSV);
        Core.inRange(inputMat, new Scalar(20, 100, 100), new Scalar(30, 255, 255), inputMat);

        //Find contours of all white pixels
        final List<MatOfPoint> points = new ArrayList<>();
        final Mat hierarchy = new Mat();
        Imgproc.findContours(inputMat, points, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        //find largest contour
        double maxArea = 0;
        int maxAreaIndex = 0;
        for (int i = 0; i < points.size(); i++) {
            final double area = Imgproc.contourArea(points.get(i));
            if (area > maxArea) {
                maxArea = area;
                maxAreaIndex = i;
            }
        }

        //draw largest contour
        final MatOfPoint largestContour = points.get(maxAreaIndex);

        //draw bounding rectangle
        final Rect boundingRect = Imgproc.boundingRect(largestContour);
        Imgproc.rectangle(outputMat, boundingRect.tl(), boundingRect.br(), new Scalar(0, 255, 0), 2);

        //draw center of bounding rectangle
        final double centerX = boundingRect.x + boundingRect.width / 2;
        final double centerY = boundingRect.y + boundingRect.height / 2;
        Imgproc.circle(outputMat, new Point(centerX, centerY), 5, new Scalar(0, 0, 255), -1);

        //draw a line for the second third of the frame that runs from top to bottom
        Imgproc.line(outputMat, new Point(outputMat.width() / 3, 0), new Point(outputMat.width() / 3, outputMat.height()), new Scalar(255, 0, 255), 2);

        //draw a bright pink line for the third third of the frame that runs from top to bottom
        Imgproc.line(outputMat, new Point(outputMat.width() * 2 / 3, 0), new Point(outputMat.width() * 2 / 3, outputMat.height()), new Scalar(255, 0, 255), 2);

        //depending on which third of the frame the rectangle is in, draw a bright orange colored number above the rectangle
        if (centerX < outputMat.width() / 3) {
            Imgproc.putText(outputMat, "0", new Point(boundingRect.x, boundingRect.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 255), 2);
            position = 0;
        } else if (centerX < outputMat.width() * 2 / 3) {
            Imgproc.putText(outputMat, "1", new Point(boundingRect.x, boundingRect.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 255), 2);
            position = 1;
        } else {
            Imgproc.putText(outputMat, "2", new Point(boundingRect.x, boundingRect.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 255), 2);
            position = 2;
        }

        //delete and recycle any unused mats
        inputMat.release();
        hierarchy.release();

        //clear memory of unused objects
        points.clear();

        return outputMat;
    }

    //function to return 0, 1, or 2 depending on the rectangle's position
    public static int getPosition() {
        return position;
    }

}
