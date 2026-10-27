variable "aws_region" {
  type        = string
  description = "AWS deployment region"
  default     = "us-east-1"
}

variable "bucket_suffix" {
  type        = string
  description = "Suffix for unique S3 bucket name"
  default     = "dev"
}
