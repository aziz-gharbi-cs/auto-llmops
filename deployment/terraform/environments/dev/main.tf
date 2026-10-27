# Development Environment Root Terraform (Section 37)
# Provisions lightweight development infrastructure boundary

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
  source      = "../../modules/networking"
  environment = "dev"
}

module "storage" {
  source        = "../../modules/storage"
  environment   = "dev"
  bucket_suffix = var.bucket_suffix
}
