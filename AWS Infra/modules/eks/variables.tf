 variable "azs" {
  description = "List of Availability Zones"
  type        = list(string)
  default     = ["us-east-1a"]  
}


variable "region" {
  type        = string
  description = "AWS Default Region"
  default     = "us-east-1"
}

variable "cluster_name" {
  type        = string
  description = "EKS cluster"
  default     = "eks-cluster"
}

variable "vpc_id" {
  type        = string
  description = "VPC ID"
}

variable "private_subnets" {
  type        = list(string)
  description = "Private Subnet CIDR values"
}

variable "control_plane_subnet_ids" {
  type        = list(string)
  description = "Control Plane Subnet IDs"  
}

variable "ami_id" {
  description = "AMI ID to use for the instance"
  type        = list(string)
}

variable "instance_type" {
  description = "Instance type to use"
  type        = list(string)
}

variable "key_name" {
  description = "Name of the key pair to use"
  type        = string
}



