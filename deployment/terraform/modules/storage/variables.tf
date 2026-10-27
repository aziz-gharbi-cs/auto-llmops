variable "environment" {
  type        = string
  description = "Target deployment environment"
}

variable "bucket_suffix" {
  type        = string
  description = "Random or organization suffix to ensure bucket uniqueness"
  default     = "bucket"
}
