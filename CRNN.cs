using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using TensorFlow;
using OpenCvSharp;
using HalconDotNet;
using System.Runtime.InteropServices;
using OpenCvSharp.Extensions;
using System.Drawing;
using System.Drawing.Imaging;
using System.ComponentModel;

namespace PAT_AOI
{
    public class CrnnOCR
    {

        TFSession sess;                                         //模型前向传播session
        TFGraph graph;

        private TFOutput imageStringPlaceholder;         //carrier图片预处理占位符
        private TFOutput imageBatchPlaceholder;    //图片batch占位符

        string inputTensorname;                            //模型一输入Tensor
        string ouputMaskname;
        string outputLabelname;                            //模型输出类别Tensor

        int[] inputTensorshape;                             //模型输入图片尺寸
        string saveNgDir;                                     //保存NG样品mask文件夹

        public string exceptionString;                             //错误类型字符串

        public bool Initial(string modelFilename, string inputTensorName, string ouputMaskName, int[] inputTensorShape)
        {
            // 初始化函数
            //  输入参数：
            //               modelFilename: 模型路径
            //               inputTensorName: 模型一输入张量名
            //               outputLabelName: 模型一输出标签张量名
            //               outputMaskName: 模型一输出Mask张量名
            //              inputTensorShape: 模型一输入张量尺寸
            //              NgDir: 保存输出Mask的路径

            graph = new TFGraph();
            inputTensorname = inputTensorName;
            ouputMaskname = ouputMaskName;
            inputTensorshape = inputTensorShape;

            var model = File.ReadAllBytes(modelFilename);       // 读取模型文件
            graph.Import(model, "");        // 从模型中导入计算图
            TFSessionOptions options = new TFSessionOptions();
            byte[] data = { 50, 09, 09, 154, 153, 153, 153, 153, 153, 201, 63 }; //0.
            IntPtr buffer = Marshal.AllocHGlobal(data.Length);
            Marshal.Copy(data, 0, buffer, data.Length);
            TFStatus ts = new TFStatus();
            options.SetConfig(buffer, data.Length, ts);
            Marshal.FreeHGlobal(buffer);
                
            //sess = new TFSession(graph);        // 创建session
            ConstructPreprocessGraph(out imageStringPlaceholder, out imageBatchPlaceholder);       //创建图片预处理计算图

            sess = new TFSession(graph, options);        // 创建session
            return true;
            

        }

        private void ConstructPreprocessGraph(out TFOutput input_1, out TFOutput output)
        {
            // 构建图像归一化，resize等预处理过程的计算图
            input_1 = graph.Placeholder(TFDataType.String);

            Double scale = 255;


            TFOutput output_1 = graph.DecodeBmp(contents: input_1, channels: inputTensorshape[2]);  // 解码
            output_1 = graph.ExpandDims(input: output_1, dim: graph.Const(0));      // 增加batch维度
            output_1 = graph.Cast(x: output_1, DstT: TFDataType.Double);     // 转类型out
            //output_1 = graph.ResizeBilinear(images: output_1, size: graph.Const(new[] { inputTensorshape[0], inputTensorshape[1] }));       // 这里的双线性插值精度为single而opencv和opcvsharp的精度为double导致resize后的图片有些许不同，如果训练时过拟合较大，则模型输出结果会因此不一样
            output_1 = graph.ResizeNearestNeighbor(images: output_1, size: graph.Const(new[] { inputTensorshape[0], inputTensorshape[1] }));    //最接近差值的精度与图片精度相同，如果为double则输出也为double

            output_1 = graph.Div(x: output_1, y: graph.Const(scale));       // 归一化
            output = graph.Cast(x: output_1, DstT: TFDataType.Float);     // 转类型
            //output = graph.ResizeNearestNeighbor(images: output_1, size: graph.Const(new[] { inputTensorshape[0], inputTensorshape[1] }));

            //output = graph.Reshape(tensor: output_1, shape: graph.Const(new[] { inputTensorshape[0], inputTensorshape[1], inputTensorshape[2] }));

        }


        public int Process(string imageFilename)
        {

            var imageBytes = File.ReadAllBytes(imageFilename);
            var imageBytesString = TFTensor.CreateString(imageBytes);
            var inputTensor = sess.Run(inputs: new[] { imageStringPlaceholder },
                                                                                    inputValues: new[] { imageBytesString },
                                                                                    outputs: new[] { imageBatchPlaceholder });     // 获取输入张量

            //Double[][][][] img = (Double[][][][])inputTensor[0].GetValue(true);
            ////Mat img = new Mat(32, 100, MatType.CV_32FC3, inputTensor[0].Data);
            ////img.ConvertTo(img, MatType.CV_8UC3, 255);

            ////Mat m = new Mat(120, 192, MatType.CV_32FC3);
            //Mat resized_m = new Mat(32, 100, MatType.CV_32FC3);
            //Mat im = Cv2.ImRead(imageFilename);
            ////im.ConvertTo(im, MatType.CV_32FC3, 1.0/255.0);
            //Cv2.Resize(im, resized_m, new OpenCvSharp.Size(100, 32));
            //resized_m.Get<Vec3f>(0, 0);

            //Cv2.ImShow("image", resized_m);
            //Cv2.WaitKey(0);


            var runner = sess.GetRunner();
            runner.AddInput(inputTensorname, inputTensor[0]);
            runner.Fetch(ouputMaskname);
            var output = runner.Run();         // 执行计算
            var raw_pred = output[0];      // mask输出

            IntPtr resData = raw_pred.Data;

            //float[][] label = (float[][])raw_pred.GetValue(true);

            int elementSize = Marshal.SizeOf(typeof(Int64));
            //int elementSize = Marshal.SizeOf(typeof(IntPtr));
            Int64 pre = 0;
            for (int i = 0; i < 25; i++)
            {

                Int64 x = Marshal.ReadInt64(resData, i * elementSize);
                //IntPtr x = Marshal.ReadIntPtr(resData, i * elementSize);
                //Console.WriteLine(x);
                if (x != pre)
                {
                    pre = x;
                    if (x != 0)
                    {
                        if (x == 10) { Console.Write(0); }
                        else if (x == 11) { Console.Write("A"); }
                        else if (x == 12) { Console.Write("B"); }
                        else if (x == 13) { Console.Write("C"); }
                        else if (x == 14) { Console.Write("D"); }
                        else if (x == 15) { Console.Write("E"); }
                        else if (x == 16) { Console.Write("F"); }
                        else if (x == 17) { Console.Write("G"); }
                        else if (x == 18) { Console.Write("H"); }
                        else if (x == 19) { Console.Write("J"); }
                        else if (x == 20) { Console.Write("K"); }
                        else if (x == 21) { Console.Write("P"); }
                        else if (x == 22) { Console.Write("S"); }
                        else if (x == 23) { Console.Write("R"); }
                        else if (x == 24) { Console.Write("Q"); }
                        else { Console.Write(x); }
                    }

                }

            }

            var image = Cv2.ImRead(imageFilename);



            Mat mean_image = emphasize(image, 15, 15, 1.5F);
            Cv2.ImShow("mean_image", mean_image);
            Cv2.ImShow("image", image);
            Cv2.WaitKey();
            Console.WriteLine("-------------------------------------------");



            //int x = resData.ToInt32();

            //var x = Marshal.PtrToStructure(resData, typeof(int));




            return 0;

        }

        Mat emphasize(Mat image, int maskHeight,int maskWidth, float factor)
        {
            Mat meanImage = new Mat(image.Size(), image.Type());

            Cv2.Blur(image, meanImage, new OpenCvSharp.Size(maskHeight, maskWidth));

            Mat empImage = image + factor*(image - meanImage);

            return empImage;

        }

    }
}