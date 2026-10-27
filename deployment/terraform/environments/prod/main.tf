# Production Environment Root Terraform (Section 37)
# Provisions full multi-AZ cluster, GPU node pools, and secure storage

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

module "networking" {
  source               = "../../modules/networking"
  environment          = "prod"
  vpc_cidr             = "10.100.0.0/16"
  public_subnet_cidrs  = ["10.100.1.0/24", "10.100.2.0/24", "10.100.3.0/24"]
  private_subnet_cidrs = ["10.100.10.0/24", "10.100.20.0/24", "10.100.30.0/24"]
  availability_zones   = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

module "storage" {
  source        = "../../modules/storage"
  environment   = "prod"
  bucket_suffix = var.bucket_suffix
}

module "kubernetes" {
  source             = "../../modules/kubernetes"
  environment        = "prod"
  subnet_ids         = module.networking.private_subnet_ids
  cluster_role_arn   = var.cluster_role_arn
  node_role_arn      = var.node_role_arn
  cpu_desired_size   = 3
  cpu_min_size       = 3
  cpu_max_size       = 15
  enable_gpu_nodes   = true
  gpu_desired_size   = 2
  gpu_min_size       = 1
  gpu_max_size       = 6
  gpu_instance_types = ["g5.2xlarge"]
}
