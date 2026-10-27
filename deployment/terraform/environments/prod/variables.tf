variable "aws_region" {
  type        = string
  description = "AWS deployment region"
  default     = "us-east-1"
}

variable "bucket_suffix" {
  type        = string
  description = "Suffix for unique S3 bucket name"
  default     = "prod"
}

variable "cluster_role_arn" {
  type        = string
  description = "IAM Role ARN for production EKS cluster"
  default     = "arn:aws:iam::123456789012:role/prod-eks-cluster-role"
}

variable "node_role_arn" {
  type        = string
  description = "IAM Role ARN for production EKS node groups"
  default     = "arn:aws:iam::123456789012:role/prod-eks-node-role"
}
