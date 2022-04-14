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
    public class DeepLearningCarrier
    {

        TFSession sess;                                         //模型前向传播session
        private TFGraph preprocessGraph;              //图片预处理计算图
        private TFSession preprocessSession;         //图片预处理session
        private TFOutput imagePlaceholder;
        private TFOutput imageBatchPlaceholder;    //图片batch占位符
        string inputTensorname;                            //模型输入Tensor名
        string outputTensorname;
        int[] inputTensorshape;                             //模型输入图片尺寸

        public string exceptionString;                             //错误类型字符串

        public bool Initial(string modelFilename, string inputTensorName, string outputLabelName, int[] inputTensorShape)
        {
            // 初始化函数
            //  输入参数：
            //               modelFilename: 模型路径
            //               inputTensorName: 模型输入张量名
            //               outputLabelName: 模型输出标签张量名

            //              inputTensorShape: 模型输入张量尺寸
            //              NgDir: 保存NG样本输出Mask的路径

            var graph = new TFGraph();
            inputTensorname = inputTensorName;
            outputTensorname = outputLabelName;
            inputTensorshape = inputTensorShape;

            try
            {
                var model = File.ReadAllBytes(modelFilename);       // 读取模型文件
                graph.Import(model, "");        // 从模型中导入计算图
                sess = new TFSession(graph);        // 创建session

                ConstructPreprocessGraph(out preprocessGraph, out imagePlaceholder, out imageBatchPlaceholder);       //创建图片预处理计算图
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

        private void ConstructPreprocessGraph(out TFGraph graph, out TFOutput input_1, out TFOutput output)
        {
            // 构建图像归一化，resize等预处理过程的计算图
            graph = new TFGraph();
            input_1 = graph.Placeholder(TFDataType.String);

            float scale = 255;
            output = graph.DecodeBmp(contents: input_1, channels: inputTensorshape[2]);  // 解码
            output = graph.Cast(x: output, DstT: TFDataType.Float);     // 转类型
            output = graph.Div(x: output, y: graph.Const(scale));       // 归一化
            output = graph.ExpandDims(input: output, dim: graph.Const(0));      // 增加batch维度
            output = graph.ResizeNearestNeighbor(images: output, size: graph.Const(new[] { inputTensorshape[0], inputTensorshape[1] }));       // resize


        }



        public int Process(string imageFilename)
        {

            int defect;

            var imageBytes = File.ReadAllBytes(imageFilename);
            var imageBytesString = TFTensor.CreateString(imageBytes);
            var inputTensor = preprocessSession.Run(inputs: new[] { imagePlaceholder },
                                                                                    inputValues: new[] { imageBytesString },
                                                                                    outputs: new[] { imageBatchPlaceholder });     // 获取输入张量

            var runner = sess.GetRunner();
            runner.AddInput(inputTensorname, inputTensor[0]);
            runner.Fetch(outputTensorname);
            var output = runner.Run();         // 执行计算
            var decision_out = output[0];      // 左mask输出

            //Int64[] label = (Int64[])decision_out.GetValue(true);
            float[][][][] label = (float[][][][])decision_out.GetValue(true);


            //Console.WriteLine(label[0]);


            return 1;

        }


        public void Dispose()
        {
            sess.Dispose();
            preprocessGraph.Dispose();
            preprocessSession.Dispose();

        }

    }

}