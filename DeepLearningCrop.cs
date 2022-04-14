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

namespace PAT_AOI
{
    public class DeepLearningProcessCrop
    {

        TFSession sess;                                         //模型前向传播session
        private TFGraph preprocessGraph;              //图片预处理计算图
        private TFSession preprocessSession;         //图片预处理session
        private TFOutput imagePlaceholder_1;         //左白光图片预处理占位符
        private TFOutput imagePlaceholder_2;         //左红光图片预处理占位符
        private TFOutput imageBatchPlaceholder;    //图片batch占位符
        string inputTensorname;                            //模型输入Tensor名
        string outputLabelname;                            //模型输出类别Tensor名
        string outputMaskname_l;                          //模型输出左Mask名
        string outputMaskname_r;                         //模型输出右Mask名
        int[] inputTensorshape;                             //模型输入图片尺寸
        string saveNgDir;                                     //保存NG样品mask文件夹

        public string exceptionString;                             //错误类型字符串

        public bool Initial(string modelFilename, string inputTensorName, string outputMaskName_l, string outputMaskName_r, int[] inputTensorShape, string NgDir)
        {
            // 初始化函数
            //  输入参数：
            //               modelFilename: 模型路径
            //               inputTensorName: 模型输入张量名
            //               outputLabelName: 模型输出标签张量名
            //               outputMaskName_l: 左图输出Mask张量名
            //               outputMaskName_r: 右图输出Mask张量名
            //              inputTensorShape: 模型输入张量尺寸
            //              NgDir: 保存NG样本输出Mask的路径

            var graph = new TFGraph();
            inputTensorname = inputTensorName;
            outputMaskname_l = outputMaskName_l;
            outputMaskname_r = outputMaskName_r;
            inputTensorshape = inputTensorShape;
            saveNgDir = NgDir;

            try
            {
                var model = File.ReadAllBytes(modelFilename);       // 读取模型文件
                graph.Import(model, "");        // 从模型中导入计算图
                sess = new TFSession(graph);        // 创建session

                ConstructPreprocessGraph(out preprocessGraph, out imagePlaceholder_1, out imagePlaceholder_2, out imageBatchPlaceholder);       //创建图片预处理计算图
                preprocessSession = new TFSession(preprocessGraph);

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

        private void ConstructPreprocessGraph(out TFGraph graph, out TFOutput input_1, out TFOutput input_2, out TFOutput output)
        {
            // 构建图像归一化，resize等预处理过程的计算图
            graph = new TFGraph();
            input_1 = graph.Placeholder(TFDataType.String);
            input_2 = graph.Placeholder(TFDataType.String);

            float scale = 255;

            TFOutput output_1 = graph.DecodeBmp(contents: input_1, channels: inputTensorshape[2]);  // 解码
            output_1 = graph.Cast(x: output_1, DstT: TFDataType.Float);     // 转类型
            output_1 = graph.Div(x: output_1, y: graph.Const(scale));       // 归一化
            output_1 = graph.ExpandDims(input: output_1, dim: graph.Const(0));      // 增加batch维度
            output_1 = graph.ResizeBilinear(images: output_1, size: graph.Const(new[] { inputTensorshape[0], inputTensorshape[1] }));       // resize

            TFOutput output_2 = graph.DecodeBmp(contents: input_2, channels: inputTensorshape[2]);  // 解码
            output_2 = graph.Cast(x: output_2, DstT: TFDataType.Float);     // 转类型
            output_2 = graph.Div(x: output_2, y: graph.Const(scale));       // 归一化
            output_2 = graph.ExpandDims(input: output_2, dim: graph.Const(0));      // 增加batch维度
            output_2 = graph.ResizeBilinear(images: output_2, size: graph.Const(new[] { inputTensorshape[0], inputTensorshape[1] }));       // resize

            output = graph.Concat(concat_dim: graph.Const(0), new TFOutput[] { output_1, output_2 });       // 将两张图片合成一个输入张量
        }



        public int Process(string imageFilename_1, string imageFilename_2)
        {
            // 处理函数，输出 0 for OK; 整数for NG classes; -1 for error, 如果NG则保存模型输出Mask
            // 输入参数：
            //              imageFilename_l: 输入左边图片
            //              imageFilename_r: 输入右边图片

            int defect;

            var imageBytes_1 = File.ReadAllBytes(imageFilename_1);
            var imageBytes_2 = File.ReadAllBytes(imageFilename_2);


            var imageBytesString_1 = TFTensor.CreateString(imageBytes_1);
            var imageBytesString_2 = TFTensor.CreateString(imageBytes_2);


            var inputTensor = preprocessSession.Run(inputs: new[] { imagePlaceholder_1, imagePlaceholder_2},
                                                                                    inputValues: new[] { imageBytesString_1, imageBytesString_2},
                                                                                    outputs: new[] { imageBatchPlaceholder });     // 获取输入张量

            var runner = sess.GetRunner();
            runner.AddInput(inputTensorname, inputTensor[0]);
            runner.Fetch(outputMaskname_l);
            runner.Fetch(outputMaskname_r);
            var output = runner.Run();         // 执行计算
            var mask_out_l = output[0];      // 左mask输出
            var mask_out_r = output[1];     // 右mask输出


            //Console.WriteLine(label[0][0]);
            //Console.WriteLine(label[1][0]);

            ////IntPtr resData1 = mask_out.Data;
            ////Mat cvmask1 = new Mat(928, 320, 5, resData1);
            ////cvmask1 = cvmask1 * 255;
            ////Cv2.ImWrite("test.png", cvmask1);

            int result = 0;
            // 如果NG则保存模型输出Mask

                //float[][][][] mask = (float[][][][])mask_out.GetValue(true);
                IntPtr resData_l = mask_out_l.Data;
                //UIntPtr dataSize = mask_out.TensorByteSize;

                //byte[] s_ImageBuffer = new byte[(int)dataSize];
                //System.Runtime.InteropServices.Marshal.Copy(resData, s_ImageBuffer, 0, (int)dataSize);

                Mat cvmask_l = new Mat(inputTensorshape[0], inputTensorshape[1], 5, resData_l);
                cvmask_l = cvmask_l * 255;
                string[] saveNgImage_l = imageFilename_1.Split(new char[] { '\\' });
                string NgFilename_l = saveNgDir + '\\' + saveNgImage_l[saveNgImage_l.Length - 1];
                Cv2.ImWrite(NgFilename_l, cvmask_l);
                result = 1;
            //Cv2.ImShow("demo", cvmask);
            //Cv2.WaitKey(0);
            //System.IO.File.WriteAllBytes("test.bmp", s_ImageBuffer);

            //byte[][][][] mask = (byte[][][][])mask_out.GetValue(true);
            //defect = PostProcessNG(NgFilename_l);


            //if (label[1][0] > 0.5)
            //{

            //    IntPtr resData_r = mask_out_r.Data;
            //    Mat cvmask_r = new Mat(inputTensorshape[0], inputTensorshape[1], 5, resData_r);
            //    cvmask_r = cvmask_r * 255;
            //    string[] saveNgImage_r = imageFilename_2.Split(new char[] { '\\' });
            //    string NgFilename_r = saveNgDir + '\\' + saveNgImage_r[saveNgImage_r.Length - 1];
            //    Cv2.ImWrite(NgFilename_r, cvmask_r);
            //    //defect =  PostProcessNG(NgFilename_r);
            //    result = 1;

            //}


            return result;



            //catch (System.IO.IOException)
            //{
            //    exceptionString = " Cannnot find image file ";
            //    return -1;

            //}
            //catch (TensorFlow.TFException)
            //{
            //    exceptionString = " Error in Inference ";
            //    return -1;

            //}
            //catch (System.ArgumentOutOfRangeException)
            //{
            //    exceptionString = " Incorrect input/output Tensorname ";
            //    return -1;
            //}

        }


        private int PostProcessNG(string maskFilename)
        {
            // TODO  对NG样本进行分类，输出1-3作为类别，出错返回-1

            //put halcon codes here
            HObject ho_Image, ho_Region;

            // Local control variables 

            HTuple hv_Area = null, hv_Row = null, hv_Column = null;
            //HTuple hv_defect = new HTuple();
            int defect = -1;
            // Initialize local and output iconic variables 
            HOperatorSet.GenEmptyObj(out ho_Image);
            HOperatorSet.GenEmptyObj(out ho_Region);
            //Image Acquisition 01: Code generated by Image Acquisition 01

            ho_Image.Dispose();
            HOperatorSet.ReadImage(out ho_Image, maskFilename);
            ho_Region.Dispose();
            HOperatorSet.Threshold(ho_Image, out ho_Region, 1, 255);

            HOperatorSet.AreaCenter(ho_Region, out hv_Area, out hv_Row, out hv_Column);
            if ((int)(new HTuple(hv_Area.TupleGreater(10000))) != 0)
            {
                //hv_defect = 0;
                defect = 1;
            }
            else if ((int)(new HTuple(hv_Area.TupleLess(10))) != 0)
            {
                //hv_defect = 1;
                defect = 2;
            }
            else
            {
                //hv_defect = 2;
                defect = 3;
            }

            ho_Image.Dispose();
            ho_Region.Dispose();

            return defect;
        }


        public void Dispose()
        {
            sess.Dispose();
            preprocessGraph.Dispose();
            preprocessSession.Dispose();

        }

    }

}

