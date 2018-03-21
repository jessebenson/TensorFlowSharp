namespace ExampleImageClassificationOnArm
{
	using System;
	using System.Collections.Generic;
	using System.IO;
	using System.Linq;
	using TensorFlow;

	/// <summary>
	/// Tensorflow image classifier.  Input is an image (as jpg encoded bytes) and
	/// runs it thorugh a Tensorflow image classification model to predict labels.
	/// </summary>
	public class ImageClassifier
	{
		private readonly TFGraph _graph;
		private readonly string[] _labels;
		private readonly string _inputName;
		private readonly string _outputName;
		private readonly int _width;
		private readonly int _height;
		private readonly float _mean;
		private readonly float _scale;

		public ImageClassifier(
			string modelFile,
			string labelFile,
			string input,
			string output,
			int width,
			int height,
			float mean,
			float scale)
		{
			// Load Tensorflow model.
			_graph = new TFGraph();
			_graph.Import(File.ReadAllBytes(modelFile));

			// Load labels.
			_labels = File.ReadAllLines(labelFile);

			// Model parameters.
			_inputName = input ?? throw new ArgumentNullException(nameof(input));
			_outputName = output ?? throw new ArgumentNullException(nameof(output));
			_width = width;
			_height = height;
			_mean = mean;
			_scale = scale;
		}

		/// <summary>
		/// Classifies the image (jpg encoded bytes).  Returns the top 2 predicted labels along with their prediction confidences.
		/// </summary>
		public ImageClassification Classify(string name, byte[] image)
		{
			// Decode the jpg image bytes.
			var tensor = CreateTensorFromImage(image);

			using (var session = new TFSession(_graph))
			{
				// Configure Tensorflow session.
				var runner = session.GetRunner()
					.AddInput(_graph[_inputName][0], tensor)
					.Fetch(_graph[_outputName][0]);

				// Run classifier.
				var output = runner.Run();
				var result = (float[,])(output[0].GetValue());

				// Return top predicted labels.
				return GetLabels(name, result);
			}
		}

		private ImageClassification GetLabels(string name, float[,] prediction)
		{
			// Join labels with predictions.
			var labels = new KeyValuePair<string, float>[_labels.Length];
			for (int i = 0; i < _labels.Length; i++)
			{
				labels[i] = new KeyValuePair<string, float>(_labels[i], prediction[0, i]);
			}

			// Return the top 2 label predictions.
			return new ImageClassification
			{
				Filename = name,
				Labels = new Dictionary<string, float>(labels.OrderByDescending(l => l.Value).Take(2)),
			};
		}

		private TFTensor CreateTensorFromImage(byte[] contents, TFDataType dataType = TFDataType.Float)
		{
			// DecodeJpeg uses a scalar String-valued tensor as input.
			var tensor = TFTensor.CreateString(contents);

			using (var graph = ConstructGraphToNormalizeImage(out TFOutput input, out TFOutput output, dataType))
			using (var session = new TFSession(graph))
			{
				var normalized = session.Run(
					inputs: new[] { input },
					inputValues: new[] { tensor },
					outputs: new[] { output });

				return normalized[0];
			}
		}

		// The inception model takes as input the image described by a Tensor in a very
		// specific normalized format (a particular image size, shape of the input tensor,
		// normalized pixel values etc.).
		//
		// This function constructs a graph of TensorFlow operations which takes as
		// input a JPEG-encoded string and returns a tensor suitable as input to the
		// inception model.
		private TFGraph ConstructGraphToNormalizeImage(out TFOutput input, out TFOutput output, TFDataType destinationDataType = TFDataType.Float)
		{
			var graph = new TFGraph();
			input = graph.Placeholder(TFDataType.String);

			output = graph.Cast(
				graph.Div(
					x: graph.Sub(
						x: graph.ResizeBilinear(
							images: graph.ExpandDims(
								input: graph.Cast(
									graph.DecodeJpeg(contents: input, channels: 3), DstT: TFDataType.Float),
								dim: graph.Const(0, "make_batch")),
							size: graph.Const(new int[] { _width, _height }, "size")),
						y: graph.Const(_mean, "mean")),
					y: graph.Const(_scale, "scale")), destinationDataType);

			return graph;
		}
	}
}
