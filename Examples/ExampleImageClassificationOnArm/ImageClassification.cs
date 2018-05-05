namespace ExampleImageClassificationOnArm
{
	using Newtonsoft.Json;
	using System;
	using System.Collections.Generic;
	using System.IO;
	using System.Linq;

	public class ImageClassification
	{
		[JsonProperty(PropertyName = "filename")]
		public string Filename { get; set; }

		[JsonProperty(PropertyName = "timestamp")]
		public DateTimeOffset Timestamp { get; set; } = DateTimeOffset.UtcNow;

		[JsonProperty(PropertyName = "labels")]
		public Dictionary<string, float> Labels { get; set; }
	}
}
