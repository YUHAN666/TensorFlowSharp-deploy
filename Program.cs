using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Threading;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace PAT_AOI
{
    class Program
    {
     

        public static void threadProcess()
        {
            DeepLearningProcess deep = new DeepLearningProcess();
            int[] inputshape2 = { 1024, 256, 1 };
            deep.Initial("./frozen_inference_graph_fuse_glue3.pb", "Image:0", "decision_out:0", "mask_out1", "mask_out2", inputshape2, "./ng2");
            List<string> files1 = Directory.GetFiles(@"E:\CODES\FAST-SCNN\DATA\surplus7\train_image\", "*.bmp").ToList();
            //List<string> files1 = Directory.GetFiles(@"F:\2\", "*.png").ToList();
            for (int i = 0; i < files1.Count; i++)
            {

                Stopwatch t = new Stopwatch();
                t.Restart();

                var result = deep.Process(files1[i], files1[i], files1[i], files1[i]);

                t.Stop();
                //Console.WriteLine(t.ElapsedMilliseconds.ToString());
                //Console.WriteLine(files1[i]);
                Console.WriteLine(result);

            }

        }

        public static void threadProcessCrop(object obj)
        {
            DeepLearningProcessCrop deep = (DeepLearningProcessCrop)obj;
            List<string> files1 = Directory.GetFiles(@"E:\CODES\FAST-SCNN\DATA\crop2\val_image\", "*.bmp").ToList();
            //List<string> files1 = Directory.GetFiles(@"F:\2\", "*.png").ToList();
            for (int i = 0; i < files1.Count; i++)
            {
                Stopwatch t = new Stopwatch();
                t.Restart();
                deep.Process(files1[i], files1[i]);
                t.Stop();
                Console.WriteLine(t.ElapsedMilliseconds.ToString());
                Console.WriteLine(files1[i]);
            }

        }

        public static void threadProcessCarrier(object obj)
        {
            DeepLearningCarrier deep = (DeepLearningCarrier)obj;
            //List<string> files1 = Directory.GetFiles(@"E:\CODES\FAST-SCNN\DATA\carrier2\", "*.bmp").ToList();
            List<string> files1 = Directory.GetFiles(@"./CARRIER2", "*.bmp").ToList();
            for (int i = 0; i < files1.Count; i++)
            {
                deep.Process(files1[i]);
                Console.WriteLine(files1[i]);
            }

        }

        public static void threadProcessYolo(object obj)
        {
            YoloProcess deep = (YoloProcess)obj;
            List<string> files1 = Directory.GetFiles(@"E:\CODES\TensorFlow_OCR\dataset\yolo_pzt\JPEGImages\", "*.jpg").ToList();
            //List<string> files1 = Directory.GetFiles(@"F:\2\", "*.png").ToList();
            for (int i = 0; i < files1.Count; i++)
            {
                Stopwatch t = new Stopwatch();
                t.Restart();
                var result = deep.Process(files1[i]);
                t.Stop();
                Console.WriteLine(t.ElapsedMilliseconds.ToString());

                Console.WriteLine(files1[i]);
                Console.WriteLine(result);
            }

        }

        public static void threadProcessOcr()
        {
            //DeepLearningOCR deep = (DeepLearningOCR)obj;
            DeepLearningOCR deep = new DeepLearningOCR();
            int[] inputshape4 = { 160, 160, 1 };
            deep.Initial("./20211007.pb", "image_input:0", "dbnet/proba3_sigmoid:0", inputshape4);
            //List<string> files1 = Directory.GetFiles(@"./carrier2/", "*.bmp").ToList();
            List<string> files1 = Directory.GetFiles(@"E:/CODES/TensorFlow_OCR/dataset/pzt_carrier/train_images/", "*.bmp").ToList();
            for (int i = 0; i < files1.Count; i++)
            {

                Stopwatch t = new Stopwatch();
                t.Restart();

                var result = deep.Process(files1[i]);
                //var reslut1 = deep.interProcess();

                t.Stop();
                //Console.WriteLine(t.ElapsedMilliseconds.ToString());
                //Console.WriteLine(files1[i]);
                //Console.WriteLine(result);

            }

        }


        public static void threadProcessCRNN()
        {
            //DeepLearningOCR deep = (DeepLearningOCR)obj;
            CrnnOCR deep = new CrnnOCR();
            int[] inputshape4 = { 32, 100, 3};
            deep.Initial("./crnn.pb", "image_input:0", "RnnHead/raw_prediction:0", inputshape4);
            //List<string> files1 = Directory.GetFiles(@"./carrier2/", "*.bmp").ToList();
            List<string> files1 = Directory.GetFiles(@"E:/CODES/TensorFlow_OCR/dataset/CRNN/test_images/", "*.bmp").ToList();
            for (int i = 0; i < files1.Count; i++)
            {

                Stopwatch t = new Stopwatch();
                t.Restart();

                var result = deep.Process(files1[i]);
                //var reslut1 = deep.interProcess();

                t.Stop();
                Console.WriteLine(t.ElapsedMilliseconds.ToString());
                //Console.WriteLine(files1[i]);
                //Console.WriteLine(result);

            }

        }

        static void Main(string[] args)
        {
            //YoloProcess deep1 = new YoloProcess();
            //DeepLearningCarrier deep1 = new DeepLearningCarrier();
            //DeepLearningProcess deep2 = new DeepLearningProcess();
            //DeepLearningProcess1 deep3 = new DeepLearningProcessCrop();

            //int[] inputshape2 = { 1024, 256, 1};

            //int[] inputshape1 = { 1024, 256, 1 };
            //deep1.Initial("./frozen_inference_graph_fuse_carrier.pb", "Image:0", "decision_out:0",  inputshape1);
            //deep1.Initial("./123.pb", "yolo_image_input:0", "yolo_image_shape:0", "yolo_input_shape:0", "yolo_output_scores:0", "yolo_output_boxes:0", "yolo_output_classes:0", inputshape1, "./ng2");
            //deep2.Initial("./frozen_inference_graph_fuse_glue3.pb", "Image:0", "decision_out:0", "mask_out1", "mask_out2", inputshape2, "./ng2");
            //deep3.Initial("./frozen_inference_graph_fuse_seg.pb", "Image:0", "mask_out1", "mask_out2", inputshape2, "./ng3");
            //DeepLearningOCR deep6 = new DeepLearningOCR();
            //DeepLearningProcess deep1 = new DeepLearningProcess();
            //DeepLearningProcess deep2 = new DeepLearningProcess();
            //DeepLearningProcess deep3 = new DeepLearningProcess();
            //DeepLearningProcess deep4 = new DeepLearningProcess();
            //DeepLearningProcess deep5 = new DeepLearningProcess();

            //int[] inputshape4 = { 160, 160, 1 };
            //deep6.Initial("./20211007.pb", "image_input:0", "dbnet/proba3_sigmoid:0", inputshape4, "./ng1");
            //deep1.Initial("./frozen_inference_graph_fuse_glue3.pb", "Image:0", "decision_out:0", "mask_out1", "mask_out2", inputshape2, "./ng2");
            //deep2.Initial("./frozen_inference_graph_fuse_glue3.pb", "Image:0", "decision_out:0", "mask_out1", "mask_out2", inputshape2, "./ng2");
            //deep3.Initial("./frozen_inference_graph_fuse_glue3.pb", "Image:0", "decision_out:0", "mask_out1", "mask_out2", inputshape2, "./ng2");
            //deep4.Initial("./frozen_inference_graph_fuse_glue3.pb", "Image:0", "decision_out:0", "mask_out1", "mask_out2", inputshape2, "./ng2");
            //deep5.Initial("./frozen_inference_graph_fuse_glue3.pb", "Image:0", "decision_out:0", "mask_out1", "mask_out2", inputshape2, "./ng2");
            //deep1.Initial("./20211007.pb", "image_input:0", "dbnet/proba3_sigmoid:0", inputshape4, "./ng1");
            //deep2.Initial("./20211007.pb", "image_input:0", "dbnet/proba3_sigmoid:0", inputshape4, "./ng1");
            //deep3.Initial("./20211007.pb", "image_input:0", "dbnet/proba3_sigmoid:0", inputshape4, "./ng1");
            //deep4.Initial("./20211007.pb", "image_input:0", "dbnet/proba3_sigmoid:0", inputshape4, "./ng1");
            //deep5.Initial("./20211007.pb", "image_input:0", "dbnet/proba3_sigmoid:0", inputshape4, "./ng1");
            

            //List<string> files1 = Directory.GetFiles(@"E:\CODES\FAST-SCNN\DATA\surplus7\val_image\", "*.bmp").ToList();

            //Stopwatch t = new Stopwatch();
            //t.Restart();
            //for (int i = 0; i < files1.Count; i++)
            //{


            //    deep1.Process(files1[i], files1[i]);
            //    deep2.Process(files1[i], files1[i]);
            //    deep3.Process(files1[i], files1[i]);

            //}
            //t.Stop();
            //Console.WriteLine(t.ElapsedMilliseconds.ToString());

            //Thread parameterizedThread1 = new Thread(new ThreadStart(threadProcess));
            //Thread parameterizedThread2 = new Thread(new ThreadStart(threadProcess));
            //Thread parameterizedThread3 = new Thread(new ThreadStart(threadProcess));
            //Thread parameterizedThread4 = new Thread(new ThreadStart(threadProcess));
           //Thread parameterizedThread5 = new Thread(new ThreadStart(threadProcess));
        
            Thread parameterizedThread6 = new Thread(new ThreadStart(threadProcessCRNN));
            //Thread parameterizedThread3 = new Thread(new ParameterizedThreadStart(threadProcess1));


            //parameterizedThread1.Start();
            //parameterizedThread2.Start();
            //parameterizedThread3.Start();
            //parameterizedThread4.Start();
            //parameterizedThread5.Start();
            parameterizedThread6.Start();
            //parameterizedThread3.Start(deep3);
            //parameterizedThread1.Join();
            //parameterizedThread1.Join();
            //parameterizedThread2.Join();
            //parameterizedThread3.Join();
            //parameterizedThread4.Join();
            //parameterizedThread5.Join();
            //parameterizedThread6.Join();
            //parameterizedThread3.Join();
            Console.Read();



        }
    }
}
