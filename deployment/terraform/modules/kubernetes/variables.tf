variable "environment" {
  type        = string
  description = "Target deployment environment"
}

variable "kubernetes_version" {
  type        = string
  description = "Kubernetes control plane version"
  default     = "1.31"
}

variable "cluster_role_arn" {
  type        = string
  description = "IAM Role ARN for EKS Cluster"
  default     = "arn:aws:iam::123456789012:role/eks-cluster-role"
}

variable "node_role_arn" {
  type        = string
  description = "IAM Role ARN for Node Groups"
  default     = "arn:aws:iam::123456789012:role/eks-node-role"
}

variable "subnet_ids" {
  type        = list(string)
  description = "List of VPC Subnet IDs for node placement"
}

variable "cpu_instance_types" {
  type        = list(string)
  description = "Instance types for CPU workloads"
  default     = ["m6i.xlarge"]
}

variable "cpu_desired_size" {
  type        = number
  default     = 2
}

variable "cpu_min_size" {
  type        = number
  default     = 1
}

variable "cpu_max_size" {
  type        = number
  default     = 10
}

variable "enable_gpu_nodes" {
  type        = bool
  description = "Whether to provision dedicated GPU inference node pool"
  default     = false
}

variable "gpu_instance_types" {
  type        = list(string)
  description = "Instance types for GPU model serving"
  default     = ["g5.xlarge"]
}

variable "gpu_desired_size" {
  type        = number
  default     = 1
}

variable "gpu_min_size" {
  type        = number
  default     = 1
}

variable "gpu_max_size" {
  type        = number
  default     = 4
}
