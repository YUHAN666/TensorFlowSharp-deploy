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
    public class DeepLearningOCR
    {
        
        TFSession sess;                                         //模型前向传播session
        private TFGraph preprocessGraph;              //图片预处理计算图
        private TFSession preprocessSession;         //图片预处理session
        private TFOutput imagePlaceholder;         //carrier图片预处理占位符
        private TFOutput imageBatchPlaceholder;    //图片batch占位符

        private TFGraph preprocessGraphDec;              //图片预处理计算图
        private TFSession preprocessSessionDec;         //图片预处理session
        private TFOutput imagePlaceholderDec;         //carrier图片预处理占位符
        private TFOutput imageBatchPlaceholderDec;    //图片batch占位符

        string inputTensorname;                            //模型一输入Tensor
        string ouputMaskname;
        string outputLabelname;                            //模型输出类别Tensor

        string inputTensornameDec = "cut_image_input:0";    //模型二输入Tensor
        string outputLabelnameDec = "decision_out:0";          //模型二输出Tensor

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

            var graph = new TFGraph();
            inputTensorname = inputTensorName;
            ouputMaskname = ouputMaskName;
            inputTensorshape = inputTensorShape;

            try
            {
                var model = File.ReadAllBytes(modelFilename);       // 读取模型文件
                graph.Import(model, "");        // 从模型中导入计算图
                TFSessionOptions options = new TFSessionOptions();
                byte[] data = { 50, 09, 09, 154, 153, 153, 153, 153, 153, 201, 63 }; //0.
                IntPtr buffer = Marshal.AllocHGlobal(data.Length);
                Marshal.Copy(data, 0, buffer, data.Length);
                TFStatus ts = new TFStatus();
                options.SetConfig(buffer, data.Length, ts);
                Marshal.FreeHGlobal(buffer);
                sess = new TFSession(graph, options);        // 创建session
                //sess = new TFSession(graph);        // 创建session
                ConstructPreprocessGraph(out preprocessGraph, out imagePlaceholder, out imageBatchPlaceholder);       //创建图片预处理计算图
                ConstructPreprocessGraphDec(out preprocessGraphDec, out imagePlaceholderDec, out imageBatchPlaceholderDec);       //创建图片预处理计算图
                preprocessSession = new TFSession(preprocessGraph);
                preprocessSessionDec = new TFSession(preprocessGraphDec);
                return true;
            }

            catch (System.IO.IOException)
            {
                exceptionString = "Cannot find modelfile";
                return false;
            }
            catch (TensorFlow.TFException)
            {
                exceptionString = " Error in Preprocessing";
                return false;
            }

        }

        private void ConstructPreprocessGraph(out TFGraph graph, out TFOutput input_1, out TFOutput output)
        {
            // 构建图像归一化，resize等预处理过程的计算图
            graph = new TFGraph();
            input_1 = graph.Placeholder(TFDataType.String);

            float scale = 255;
     

            TFOutput output_1 = graph.DecodeBmp(contents: input_1, channels: inputTensorshape[2]);  // 解码
            output_1 = graph.Cast(x: output_1, DstT: TFDataType.Float);     // 转类型
            output_1 = graph.Div(x: output_1, y: graph.Const(scale));       // 归一化


            output_1 = graph.ExpandDims(input: output_1, dim: graph.Const(0));      // 增加batch维度

            output_1 = graph.ResizeBilinear(images: output_1, size: graph.Const(new[] {100, inputTensorshape[0] }));       // resize 100=1200/(1920 / inputTensorshape[0])
            output_1 = graph.Pad(input: output_1, paddings: graph.Const(new[,] { { 0, 0 }, { 0, 60 }, { 0, 0 }, { 0, 0 } }));  //pad to (inputTensorshape[0], inputTensorshape[0])， 60=inputTensorshape[0]-100

            output = graph.Concat(concat_dim: graph.Const(3), new TFOutput[] { output_1, output_1, output_1 });       // 合成三通道图片
        }

        private void ConstructPreprocessGraphDec(out TFGraph graph, out TFOutput input_1, out TFOutput output)
        {
            // 构建图像归一化，resize等预处理过程的计算图
            graph = new TFGraph();
            input_1 = graph.Placeholder(TFDataType.String);

            float scale = 255;


            TFOutput output_1 = graph.DecodeBmp(contents: input_1, channels: 1);  // 解码
            output_1 = graph.Cast(x: output_1, DstT: TFDataType.Float);     // 转类型
            output_1 = graph.Div(x: output_1, y: graph.Const(scale));       // 归一化


            output_1 = graph.ExpandDims(input: output_1, dim: graph.Const(0));      // 增加batch维度

            output_1 = graph.ResizeBilinear(images: output_1, size: graph.Const(new[] { 96, 96 }));       

            output = graph.Concat(concat_dim: graph.Const(3), new TFOutput[] { output_1, output_1, output_1 });       // 三通道

        }


        public Bitmap HObject2Bitmap(HObject ho_Image)  //单通道图像从Halcon object转bitmap (图像尺寸96*96)
        {
            HTuple  hv_Pointer = new HTuple(), hv_Height = new HTuple();
            HTuple hv_Type = new HTuple(), hv_Width = new HTuple();
            HOperatorSet.GetImagePointer1(ho_Image, out hv_Pointer, out hv_Type, out hv_Width, out hv_Height);
            //byte[] gray_image = new byte[1920 * 1200];
            //Marshal.Copy(hv_Pointer, gray_image, 0, 1920 * 1200);
            //Mat img = new Mat(1200, 1920, 0, gray_image);
            byte[] gray_image = new byte[96 * 96];
            Marshal.Copy(hv_Pointer, gray_image, 0, 96 * 96);
            

            Mat img = new Mat(96, 96, 0, gray_image);
            //Cv2.ImShow("image", img);
            //Cv2.WaitKey();

            Bitmap bitmap = BitmapConverter.ToBitmap(img);
            return bitmap;
        }

        private Mat HObject2cvMat(HObject ho_Image)
        {
            HTuple hv_Pointer = new HTuple(), hv_Height = new HTuple();
            HTuple hv_Type = new HTuple(), hv_Width = new HTuple();
            HOperatorSet.GetImagePointer1(ho_Image, out hv_Pointer, out hv_Type, out hv_Width, out hv_Height);
            byte[] gray_image = new byte[1920 * 1200];
            Marshal.Copy(hv_Pointer, gray_image, 0, 1920 * 1200);
            Mat img = new Mat(1200, 1920, 0, gray_image);
            return img;

        }

        public byte[] Bitmap2Byte(Bitmap bitmap)    // bitmap类型图片转二进制流(含bmp头)给tensorflow DecodeBmp使用
        {
            MemoryStream Ms = new MemoryStream();
            
            bitmap.Save(Ms, ImageFormat.Bmp);
            byte[] imageByte = new byte[Ms.Length];

            Ms.Position = 0;
            Ms.Read(imageByte, 0, Convert.ToInt32(Ms.Length));
            Ms.Close();

            return imageByte;

        }
        public string Process(string imageFilename)
        {

            var imageBytes = File.ReadAllBytes(imageFilename);      /// TODO:读取carrier照片
            var imageBytesString = TFTensor.CreateString(imageBytes);
            var inputTensor = preprocessSession.Run(inputs: new[] { imagePlaceholder },
                                                                                    inputValues: new[] { imageBytesString },
                                                                                    outputs: new[] { imageBatchPlaceholder });     // 获取输入张量

            var runner = sess.GetRunner();
            runner.AddInput(inputTensorname, inputTensor[0]);
            runner.Fetch(ouputMaskname);
            var output = runner.Run();         // 执行计算
            var mask_out = output[0];      // mask输出
            

            IntPtr resData = mask_out.Data;
            Mat cvmask_l = new Mat(inputTensorshape[0], inputTensorshape[1], 5, resData);
            cvmask_l = cvmask_l * 255;
            string[] saveNgImage_l = imageFilename.Split(new char[] { '\\' });
            string NgFilename_l = saveNgDir + '\\' + saveNgImage_l[saveNgImage_l.Length - 1];
            //Cv2.ImWrite(NgFilename_l, cvmask_l);
            //Cv2.ImShow("image", cvmask_l);
            //Cv2.WaitKey();


            var ocr_result = interProcess(cvmask_l, imageFilename);     //使用第一个模型输出的mask切割字符，并输入第二个模型得到结果
            string result_str = "";
            for (int i=0; i< ocr_result.Count; i++)
                result_str += ocr_result[i].ToString();
            //Mat resized_img = new Mat(320,320,0);
            //Cv2.Resize(img, resized_img, new OpenCvSharp.Size(960, 960));
            //Cv2.PutText(resized_img, result_str, new OpenCvSharp.Point(300, 200), HersheyFonts.HersheySimplex, 2, new Scalar(0,0,255));
            //Cv2.ImShow("image", resized_img);
            //Cv2.WaitKey();
            string s = imageFilename.Split('/').Last().Split('-')[0];
            if (s != result_str && (("00" + s) != result_str) && (("0" + s) != result_str)) 
            {
                Console.WriteLine(imageFilename);
                Console.WriteLine(result_str);
            }
               
            return result_str;
           

        }


        public void cut_image(HObject ho_image, HObject ho_region, out HObject ho_image_cut)
        {



            // Local control variables 

            HTuple hv_Row1 = null, hv_Column1 = null, hv_Row2 = null;
            HTuple hv_Column2 = null;
            // Initialize local and output iconic variables 
            HOperatorSet.GenEmptyObj(out ho_image_cut);
            HOperatorSet.SmallestRectangle1(ho_region, out hv_Row1, out hv_Column1, out hv_Row2,
                out hv_Column2);
            ho_image_cut.Dispose();
            HOperatorSet.CropRectangle1(ho_image, out ho_image_cut, hv_Row1, hv_Column1,
                hv_Row2, hv_Column2);

            return;
        }

        public List<Int64> interProcess(Mat mask, string filename) 
            // 使用第一个模型输出的mask切割字符，并输入第二个模型得到结果
            // filename: 第一个模型输入的文件名
            // mask: 第一个模型输出的结果
        {
            Bitmap bitmap;
            byte[] imageBytes=null;
            List<Int64> result = new List<Int64>();

            HObject ho_ImageOri = null, ho_Image = null, ho_Region = null;
            HObject ho_RegionOpening = null, ho_RegionClosing = null, ho_ConnectedRegions = null;
            HObject ho_SortedRegions = null, ho_ObjectSelected = null, ho_Rectangle1 = null;
            HObject ho_image_cut = null, ho_Rectangle2 = null, ho_ImageZoom = null;
            HObject ho_Rectangle3 = null, ho_Rectangle4 = null;
            HImage image = new HImage();
            // Local control variables 

            HTuple hv_WIDTH = null, hv_HEIGHT = null, hv_SCALE = null;
            HTuple hv_save_root = null, hv_ImageFiles = null, hv_ImageFiles1 = null;
            HTuple hv_Index = null, hv_path = new HTuple(), hv_Number1 = new HTuple();
            HTuple hv_i = new HTuple(), hv_Area = new HTuple(), hv_Row1 = new HTuple();
            HTuple hv_Column1 = new HTuple(), hv_name_str = new HTuple();
            HTuple hv_Area3 = new HTuple(), hv_Row3 = new HTuple();
            HTuple hv_Column3 = new HTuple(), hv_Pointer = new HTuple();
            HTuple hv_Type = new HTuple(), hv_Width = new HTuple();
            HTuple hv_Height = new HTuple();
            // Initialize local and output iconic variables 
            HOperatorSet.GenEmptyObj(out ho_ImageOri);
            HOperatorSet.GenEmptyObj(out ho_Image);
            HOperatorSet.GenEmptyObj(out ho_Region);
            HOperatorSet.GenEmptyObj(out ho_RegionOpening);
            HOperatorSet.GenEmptyObj(out ho_RegionClosing);
            HOperatorSet.GenEmptyObj(out ho_ConnectedRegions);
            HOperatorSet.GenEmptyObj(out ho_SortedRegions);
            HOperatorSet.GenEmptyObj(out ho_ObjectSelected);
            HOperatorSet.GenEmptyObj(out ho_Rectangle1);
            HOperatorSet.GenEmptyObj(out ho_image_cut);
            HOperatorSet.GenEmptyObj(out ho_Rectangle2);
            HOperatorSet.GenEmptyObj(out ho_ImageZoom);
            HOperatorSet.GenEmptyObj(out ho_Rectangle3);
            HOperatorSet.GenEmptyObj(out ho_Rectangle4);
            hv_WIDTH = 360;         //原图尺寸下，字符块的宽
            hv_HEIGHT = 440;        //原图尺寸下，字符块的高
            if (HDevWindowStack.IsOpen())
            {
                HOperatorSet.SetDraw(HDevWindowStack.GetActive(), "margin");
            }
            hv_SCALE = 12;      // 12=1920/inputTensorshape[0]

            ho_ImageOri.Dispose();
                
            HOperatorSet.ReadImage(out ho_ImageOri, filename);      /// TODO: 需要将Process的图片读取为Halcon object
            ho_Image.Dispose();
            ho_Image = cvMat2Hobject(mask);         //将opencv格式图片转化为halcon object
            ho_Region.Dispose();
            HOperatorSet.Threshold(ho_Image, out ho_Region, 125, 255);

            ho_RegionOpening.Dispose();
            HOperatorSet.OpeningRectangle1(ho_Region, out ho_RegionOpening, 5, 5);
            ho_RegionClosing.Dispose();
            HOperatorSet.ClosingRectangle1(ho_RegionOpening, out ho_RegionClosing, 5, 5);
            if (HDevWindowStack.IsOpen())
            {
                HOperatorSet.ClearWindow(HDevWindowStack.GetActive());
            }

            ho_ConnectedRegions.Dispose();
            HOperatorSet.Connection(ho_RegionClosing, out ho_ConnectedRegions);


            HOperatorSet.CountObj(ho_ConnectedRegions, out hv_Number1);


            if ((int)(new HTuple(hv_Number1.TupleLess(3))) != 0)
            {
                //识别到少于三个字符块（有字符连到一起了）
                ho_SortedRegions.Dispose();
                HOperatorSet.SortRegion(ho_ConnectedRegions, out ho_SortedRegions, "upper_left",
                    "true", "column");
                HTuple end_val26 = hv_Number1;
                HTuple step_val26 = 1;
                for (hv_i = 1; hv_i.Continue(end_val26, step_val26); hv_i = hv_i.TupleAdd(step_val26))
                {
                    ho_ObjectSelected.Dispose();
                    HOperatorSet.SelectObj(ho_SortedRegions, out ho_ObjectSelected, hv_i);
                    HOperatorSet.AreaCenter(ho_ObjectSelected, out hv_Area, out hv_Row1, out hv_Column1);
                    if ((int)(new HTuple(hv_Area.TupleGreater(4000))) != 0)
                    {
                        //面积大于4000说明是连到一起的两个字符，需要分开
                        ho_Rectangle1.Dispose();
                        HOperatorSet.GenRectangle1(out ho_Rectangle1, (hv_Row1 * hv_SCALE) - (0.5 * hv_HEIGHT),
                            (hv_Column1 * hv_SCALE) - hv_WIDTH, (hv_Row1 * hv_SCALE) + (0.5 * hv_HEIGHT),
                            hv_Column1 * hv_SCALE);
                        ho_image_cut.Dispose();
                        cut_image(ho_ImageOri, ho_Rectangle1, out ho_image_cut);
                        //HOperatorSet.WriteImage(ho_image_cut, "jpeg", 0, hv_name_str);
                        ho_ImageZoom.Dispose();
                        HOperatorSet.ZoomImageSize(ho_image_cut, out ho_ImageZoom, 96, 96, "bilinear");
                        //HOperatorSet.WriteImage(ho_ImageZoom, "jpeg", 0, hv_name_str);
                        bitmap = HObject2Bitmap(ho_ImageZoom);
                        imageBytes = Bitmap2Byte(bitmap);


                        var imageBytesStringDec = TFTensor.CreateString(imageBytes);


                        var inputTensorDec = preprocessSessionDec.Run(inputs: new[] { imagePlaceholderDec },
                                                                                                inputValues: new[] { imageBytesStringDec },
                                                                                                outputs: new[] { imageBatchPlaceholderDec });     // 模型2获取输入张量

                        var runnerDec = sess.GetRunner();
                        runnerDec.AddInput(inputTensornameDec, inputTensorDec[0]);
                        runnerDec.Fetch(outputLabelnameDec);
                        var outputDec = runnerDec.Run();         // 模型2执行计算
                        var decision_out = outputDec[0];
                        //Int64[] label = (Int64[])decision_out.GetValue(true);
                        Int64[] label = (Int64[])decision_out.GetValue(true);
                        result.Add(label[0]);


                        ho_Rectangle2.Dispose();
                        HOperatorSet.GenRectangle1(out ho_Rectangle2, (hv_Row1 * hv_SCALE) - (0.5 * hv_HEIGHT),
                            hv_Column1 * hv_SCALE, (hv_Row1 * hv_SCALE) + (0.5 * hv_HEIGHT), (hv_Column1 * hv_SCALE) + hv_WIDTH);
                        ho_image_cut.Dispose();
                        cut_image(ho_ImageOri, ho_Rectangle2, out ho_image_cut);
                        ho_ImageZoom.Dispose();
                        HOperatorSet.ZoomImageSize(ho_image_cut, out ho_ImageZoom, 96, 96, "bilinear");
                        //HOperatorSet.WriteImage(ho_ImageZoom, "jpeg", 0, hv_name_str);
                        bitmap = HObject2Bitmap(ho_ImageZoom);
                        imageBytes = Bitmap2Byte(bitmap);


                        var imageBytesStringDec1 = TFTensor.CreateString(imageBytes);


                        var inputTensorDec1 = preprocessSessionDec.Run(inputs: new[] { imagePlaceholderDec },
                                                                                                inputValues: new[] { imageBytesStringDec },
                                                                                                outputs: new[] { imageBatchPlaceholderDec });     // 模型2获取输入张量

                        var runnerDec1 = sess.GetRunner();
                        runnerDec1.AddInput(inputTensornameDec, inputTensorDec1[0]);
                        runnerDec1.Fetch(outputLabelnameDec);
                        var outputDec1 = runnerDec1.Run();         // 模型2执行计算
                        var decision_out1 = outputDec1[0];
                        //Int64[] label = (Int64[])decision_out.GetValue(true);
                        Int64[] label1 = (Int64[])decision_out1.GetValue(true);

                        result.Add(label1[0]);

                    }
                    else
                    {
                        ho_Rectangle3.Dispose();
                        HOperatorSet.GenRectangle1(out ho_Rectangle3, (hv_Row1 * hv_SCALE) - (0.5 * hv_HEIGHT),
                            (hv_Column1 * hv_SCALE) - (0.5 * hv_WIDTH), (hv_Row1 * hv_SCALE) + (0.5 * hv_HEIGHT),
                            (hv_Column1 * hv_SCALE) + (0.5 * hv_WIDTH));
                        ho_image_cut.Dispose();
                        cut_image(ho_ImageOri, ho_Rectangle3, out ho_image_cut);
                        //hv_name_str = (((hv_save_root + hv_Index) + "_") + hv_i) + ".jpg";
                        ho_ImageZoom.Dispose();
                        HOperatorSet.ZoomImageSize(ho_image_cut, out ho_ImageZoom, 96, 96, "bilinear");
                        // HOperatorSet.WriteImage(ho_ImageZoom, "jpeg", 0, hv_name_str);
                        bitmap = HObject2Bitmap(ho_ImageZoom);
                        imageBytes = Bitmap2Byte(bitmap);

                        var imageBytesStringDec = TFTensor.CreateString(imageBytes);


                        var inputTensorDec = preprocessSessionDec.Run(inputs: new[] { imagePlaceholderDec },
                                                                                                inputValues: new[] { imageBytesStringDec },
                                                                                                outputs: new[] { imageBatchPlaceholderDec });     // 获取输入张量

                        var runnerDec = sess.GetRunner();
                        runnerDec.AddInput(inputTensornameDec, inputTensorDec[0]);
                        runnerDec.Fetch(outputLabelnameDec);
                        var outputDec = runnerDec.Run();         // 执行计算
                        var decision_out = outputDec[0];
                        //Int64[] label = (Int64[])decision_out.GetValue(true);
                        Int64[] label = (Int64[])decision_out.GetValue(true);

                        //Mat img = Cv2.ImRead("E:/DATA/543.bmp");
                        //Mat img = new Mat(96, 96, 0, imageBytes);
                        //Cv2.ImWrite("E:/DATA/4321.jpg", img);
                        //Cv2.ImShow("image", img);
                        //Cv2.WaitKey();
                        result.Add(label[0]);


                    }
                }

            }
            else
            {
                ho_SortedRegions.Dispose();
                HOperatorSet.SortRegion(ho_ConnectedRegions, out ho_SortedRegions, "upper_left",
                    "true", "column");
                HTuple end_val51 = hv_Number1;
                HTuple step_val51 = 1;
                for (hv_i = 1; hv_i.Continue(end_val51, step_val51); hv_i = hv_i.TupleAdd(step_val51))
                {
                    ho_ObjectSelected.Dispose();
                    HOperatorSet.SelectObj(ho_SortedRegions, out ho_ObjectSelected, hv_i);
                    HOperatorSet.AreaCenter(ho_ObjectSelected, out hv_Area3, out hv_Row3, out hv_Column3);
                    ho_Rectangle4.Dispose();
                    HOperatorSet.GenRectangle1(out ho_Rectangle4, (hv_Row3 * hv_SCALE) - (0.5 * hv_HEIGHT),
                        (hv_Column3 * hv_SCALE) - (0.5 * hv_WIDTH), (hv_Row3 * hv_SCALE) + (0.5 * hv_HEIGHT),
                        (hv_Column3 * hv_SCALE) + (0.5 * hv_WIDTH));

                    ho_image_cut.Dispose();
                    cut_image(ho_ImageOri, ho_Rectangle4, out ho_image_cut);
                    ho_ImageZoom.Dispose();
                    HOperatorSet.ZoomImageSize(ho_image_cut, out ho_ImageZoom, 96, 96,  "constant");
                    bitmap = HObject2Bitmap(ho_ImageZoom);

                    //bitmap.Save("E:/DATA/543.bmp", ImageFormat.Bmp);
                    imageBytes = Bitmap2Byte(bitmap);

                    var imageBytesStringDec = TFTensor.CreateString(imageBytes);


                    var inputTensorDec = preprocessSessionDec.Run(inputs: new[] { imagePlaceholderDec },
                                                                                            inputValues: new[] { imageBytesStringDec },
                                                                                            outputs: new[] { imageBatchPlaceholderDec });     // 模型2获取输入张量

                    var runnerDec = sess.GetRunner();
                    runnerDec.AddInput(inputTensornameDec, inputTensorDec[0]);
                    runnerDec.Fetch(outputLabelnameDec);
                    var outputDec = runnerDec.Run();         // 模型2执行计算
                    var decision_out = outputDec[0];
                    //Int64[] label = (Int64[])decision_out.GetValue(true);
                    Int64[] label = (Int64[])decision_out.GetValue(true);

                    //Mat img = Cv2.ImRead("E:/DATA/543.bmp");
                    //Mat img = new Mat(96, 96, 0, imageBytes);
                    //Cv2.ImWrite("E:/DATA/4321.jpg", img);
                    //Cv2.ImShow("image", img);
                    //Cv2.WaitKey();
                    result.Add(label[0]);

                }
            }

            ho_ImageOri.Dispose();
            ho_Image.Dispose();
            ho_Region.Dispose();
            ho_RegionOpening.Dispose();
            ho_RegionClosing.Dispose();
            ho_ConnectedRegions.Dispose();
            ho_SortedRegions.Dispose();
            ho_ObjectSelected.Dispose();
            ho_Rectangle1.Dispose();
            ho_image_cut.Dispose();
            ho_Rectangle2.Dispose();
            ho_ImageZoom.Dispose();
            ho_Rectangle3.Dispose();
            ho_Rectangle4.Dispose();

            return result;
        }

        private void HobjectToHimage(HObject hobject, ref HImage image)
        {
            HTuple pointer, type, width, height;
            HOperatorSet.GetImagePointer1(hobject, out pointer, out type, out width, out height);
            image.GenImage1(type, width, height, pointer);
        }

        private HObject cvMat2Hobject(Mat img)
        {
            HObject image = null;
            HTuple hv_Height = new HTuple(160);     //160是模型输出mask尺寸
            HTuple hv_Type = new HTuple("byte");
            HTuple hv_Width = new HTuple(160);
            img.ConvertTo(img, 0);      //从32F转8U
            IntPtr ptr = img.Data;

            HOperatorSet.GenImage1(out image, hv_Type, hv_Width, hv_Height, ptr);

            return image;

        }

    }
}