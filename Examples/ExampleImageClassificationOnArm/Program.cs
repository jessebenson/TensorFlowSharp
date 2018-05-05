namespace ExampleImageClassificationOnArm
{
    using System;
    using System.IO;
    using System.Linq;
    using System.Threading;
    using System.Threading.Tasks;
    using Newtonsoft.Json;
    using TensorFlow;

    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine ("TensorFlow version: " + TFCore.Version);

            var classifier = new ImageClassifier(
                modelFile: "model/model.pb",
                labelFile: "model/labels.txt",
                input: "input",
                output: "final_result",
                width: 224,
                height: 224,
                mean: 128,
                scale: 255);

            // Run image classification on each file.
            var images = new [] { "images/iris-flower.jpg", "images/rose-flower.jpg", "images/tulip-flower.jpg" };
            foreach (var fileName in images)
            {
                byte[] imageBytes = File.ReadAllBytes(fileName);
                ImageClassification result = classifier.Classify(fileName, imageBytes);

                // Display the top prediction.
                var prediction = result.Labels.OrderByDescending(kvp => kvp.Value).First();
                Console.WriteLine($"Classification prediction for {fileName}: {prediction.Key} with probability {prediction.Value:0.00}");
            }
        }
    }
}
